"""Tests: discarding incomplete answers during grading and re-running.
Run with:  python3 -m unittest test_hub_discard.py"""

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
    from hub_client import encode_multipart


@unittest.skipUnless(FLASK, "Flask is not installed")
class DiscardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        db = connect(self.app.config["DATABASE"])
        self.pi_id, _ = create_user(db, "PI", "pi@example.org", "pi", invited=False)
        self.g1, _ = create_user(db, "Alice", "a@example.org", "grader",
                                 invited=False)
        self.g2, _ = create_user(db, "Bob", "b@example.org", "grader",
                                 invited=False)
        for user_id, password in ((self.pi_id, "pi-password"),
                                  (self.g1, "alice-pass"), (self.g2, "bob-pass")):
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                       (generate_password_hash(password), user_id))
        db.execute(
            "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
            "VALUES ('G001-001', ?, 1, 'Case.', ?)",
            (self.g1, json.dumps(["r1"])),
        )
        db.execute(
            "UPDATE users SET max_assigned_case_number = 1 WHERE id = ?",
            (self.g1,),
        )
        # One run job Alice ran herself, with a truncated scraped answer.
        db.execute(
            "INSERT INTO run_jobs (requested_by, assigned_to, case_ids, llm_ids) "
            "VALUES (?, ?, ?, ?)",
            (self.g1, self.g1, json.dumps(["G001-001"]),
             json.dumps(["openevidence"])),
        )
        self.token = "alice-runner"
        db.execute("INSERT INTO runner_tokens (user_id, token_hash) VALUES (?, ?)",
                   (self.g1, hash_token(self.token)))
        db.commit()
        db.close()
        self.upload("The answer was cut off mid-sen")

    def tearDown(self):
        self.tmp.cleanup()

    def upload(self, text):
        body, content_type = encode_multipart({
            "run_job_id": "1", "case_id": "G001-001",
            "llm_id": "openevidence", "model_name": "OE 2026-07",
            "variant_id": "openevidence@oe-2026-07",
            "model_display_name": "OpenEvidence (OE 2026-07)",
            "response_text": text, "rubric_version": "1",
        }, [])
        return self.client.post(
            "/api/runner/answers", data=body, content_type=content_type,
            headers={"Authorization": "Bearer " + self.token},
        )

    def login(self, email="a@example.org", password="alice-pass"):
        self.client.get("/logout")
        return self.client.post("/login", data={"email": email,
                                                "password": password})

    def db(self):
        return connect(self.app.config["DATABASE"])

    def cross_assign_bob(self):
        self.login("pi@example.org", "pi-password")
        self.client.post("/assignments", data={
            "grader_id": str(self.g2), "case_id": ["G001-001"],
        })

    def test_discard_hides_voids_and_reports_for_rerun(self):
        self.cross_assign_bob()
        # Both graders grade the truncated answer first.
        self.login("b@example.org", "bob-pass")
        self.client.get("/grade")
        self.client.post("/grade/G001-001/A", data={"item_0": "no"})
        self.login()
        self.client.get("/grade")
        self.client.post("/grade/G001-001/A", data={"item_0": "no"})
        # Alice notices it is cut off and discards it.
        response = self.client.post("/grade/G001-001/A/discard", data={
            "reason": "cut off mid-sentence",
        }, follow_redirects=True)
        self.assertIn(b"was discarded", response.data)
        self.assertIn(b"2 existing grade(s) on it were set aside", response.data)
        # Gone from Alice's pages...
        self.assertEqual(self.client.get("/grade/G001-001/A").status_code, 404)
        page = self.client.get("/grade/G001-001")
        self.assertNotIn(b">A<", page.data)
        # ...and from Bob's.
        self.login("b@example.org", "bob-pass")
        self.assertEqual(self.client.get("/grade/G001-001/A").status_code, 404)
        # All grades on it are superseded; the rankings see nothing.
        db = self.db()
        active = db.execute(
            "SELECT COUNT(*) AS n FROM grades WHERE superseded = 0"
        ).fetchone()["n"]
        db.close()
        self.assertEqual(active, 0)
        self.login("pi@example.org", "pi-password")
        self.assertIn(b"No usable grades yet", self.client.get("/rankings").data)
        # The owner's Run jobs page lists it for a re-run.
        self.login()
        page = self.client.get("/runs")
        self.assertIn(b"needs a re-run", page.data)
        self.assertIn(b"cut off mid-sentence", page.data)
        self.assertIn(b"OpenEvidence (OE 2026-07)", page.data)

    def test_rerun_revives_the_same_slot_for_fresh_grading(self):
        self.login()
        self.client.get("/grade")
        self.client.post("/grade/G001-001/A/discard", data={"reason": "junk"})
        db = self.db()
        old_id = db.execute("SELECT id FROM answers").fetchone()["id"]
        db.close()
        # A fresh run (same case, same variant, same runner) uploads.
        self.upload("The complete answer, captured properly this time.")
        db = self.db()
        row = db.execute("SELECT * FROM answers").fetchone()
        db.close()
        self.assertEqual(row["id"], old_id)          # same slot revived
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["discarded_by"])
        self.assertEqual(row["discarded_reason"], "")
        # Same letter, back in the queue, gradable again.
        page = self.client.get("/grade/G001-001")
        self.assertIn(b"not graded", page.data)
        response = self.client.post("/grade/G001-001/A", data={
            "item_0": "yes", "risk": "no", "poor": "no",
        }, follow_redirects=True)
        self.assertIn(b"scored 2", response.data)
        self.assertIn(b"captured properly",
                      self.client.get("/grade/G001-001/A").data)
        # The re-run card is gone.
        self.assertNotIn(b"needs a re-run", self.client.get("/runs").data)

    def test_only_assigned_graders_can_discard(self):
        self.login()
        self.client.get("/grade")  # creates Alice's labels
        self.login("b@example.org", "bob-pass")  # Bob has NO assignment
        response = self.client.post("/grade/G001-001/A/discard",
                                    data={"reason": "x"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
