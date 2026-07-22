"""Tests: the PI can also act as a grader (one account, both hats).
Run with:  python3 -m unittest test_hub_pi_grader.py"""

import json
import tempfile
import unittest

try:
    import flask  # noqa: F401

    FLASK = True
except ImportError:
    FLASK = False

if FLASK:
    from werkzeug.security import generate_password_hash

    from hub import create_app
    from hub.auth import create_user
    from hub.db import connect
    from hub.runner_api import hash_token


@unittest.skipUnless(FLASK, "Flask is not installed")
class PiAsGraderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        db = connect(self.app.config["DATABASE"])
        self.pi_id, _ = create_user(db, "PI", "pi@example.org", "pi", invited=False)
        self.g1, _ = create_user(db, "Alice", "a@example.org", "grader",
                                 invited=False)
        for user_id, password in ((self.pi_id, "pi-password"),
                                  (self.g1, "alice-pass")):
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                       (generate_password_hash(password), user_id))
        db.execute(
            "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
            "VALUES ('G001-001', ?, 1, 'Alice case.', ?)",
            (self.g1, json.dumps(["r1"])),
        )
        db.execute(
            "UPDATE users SET max_assigned_case_number = 1 WHERE id = ?",
            (self.g1,),
        )
        self.pi_token = "pi-runner-token"
        db.execute("INSERT INTO runner_tokens (user_id, token_hash) VALUES (?, ?)",
                   (self.pi_id, hash_token(self.pi_token)))
        db.commit()
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def login(self, email="pi@example.org", password="pi-password"):
        self.client.get("/logout")
        return self.client.post("/login", data={"email": email,
                                                "password": password})

    def db(self):
        return connect(self.app.config["DATABASE"])

    def test_first_new_case_makes_the_pi_a_grader(self):
        self.login()
        # Before opting in, the Grade page redirects to assignments.
        response = self.client.get("/grade")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/assignments", response.headers["Location"])
        response = self.client.post("/cases/new", data={
            "case_text": "PI's own case.", "rubric": "item 1",
        }, follow_redirects=True)
        self.assertIn(b"You are now also grader 2", response.data)
        self.assertIn(b"G002-001", response.data)
        # The PI can edit and delete their own case...
        response = self.client.post("/cases/G002-001", data={
            "case_text": "PI's own case, edited.", "rubric": "item 1",
        }, follow_redirects=True)
        self.assertIn(b"Saved case G002-001", response.data)
        # ...but still cannot delete Alice's.
        response = self.client.post("/cases/G001-001/delete")
        self.assertEqual(response.status_code, 404)

    def test_pi_runs_and_grades_their_own_case_end_to_end(self):
        self.login()
        self.client.post("/cases/new", data={
            "case_text": "PI case.", "rubric": "item 1",
        })
        # Create a run job for themselves and answer it via the runner API.
        response = self.client.post("/runs", data={
            "case_id": ["G002-001"], "llm_id": ["testmodel"], "assignee": "me",
        }, follow_redirects=True)
        self.assertIn(b"Run job created", response.data)
        jobs = self.client.get(
            "/api/runner/jobs",
            headers={"Authorization": "Bearer " + self.pi_token},
        ).get_json()["jobs"]
        self.assertEqual(len(jobs), 1)
        from hub_client import encode_multipart

        body, content_type = encode_multipart({
            "run_job_id": str(jobs[0]["id"]), "case_id": "G002-001",
            "llm_id": "testmodel", "variant_id": "testmodel@test-model-1",
            "model_display_name": "Test model", "response_text": "answer",
            "rubric_version": "1",
        }, [])
        self.client.post("/api/runner/answers", data=body,
                         content_type=content_type,
                         headers={"Authorization": "Bearer " + self.pi_token})
        # Grade it blinded.
        page = self.client.get("/grade")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"G002-001", page.data)
        response = self.client.post("/grade/G002-001/A", data={
            "item_0": "yes", "risk": "no", "poor": "no",
        }, follow_redirects=True)
        self.assertIn(b"scored 2", response.data)
        # Their grade reaches My results and the rankings.
        page = self.client.get("/my-results")
        self.assertIn(b"Test model", page.data)
        page = self.client.get("/rankings")
        self.assertIn(b"1 graded answers", page.data)

    def test_pi_appears_in_the_cross_grading_dropdown(self):
        self.login()
        self.client.post("/cases/new", data={
            "case_text": "PI case.", "rubric": "item 1",
        })
        page = self.client.get("/assignments")
        self.assertIn(b"PI", page.data)
        # And the PI can be assigned Alice's case for cross-grading.
        response = self.client.post("/assignments", data={
            "grader_id": str(self.pi_id), "case_id": ["G001-001"],
        }, follow_redirects=True)
        self.assertIn(b"Assigned 1 case(s) to PI", response.data)

    def test_regular_graders_are_unchanged(self):
        self.login("a@example.org", "alice-pass")
        response = self.client.post("/cases/new", data={
            "case_text": "Another Alice case.", "rubric": "r",
        }, follow_redirects=True)
        self.assertIn(b"G001-002", response.data)
        self.assertNotIn(b"You are now also grader", response.data)
        self.assertEqual(self.client.get("/rankings").status_code, 403)


if __name__ == "__main__":
    unittest.main()
