"""Tests for the Study Hub phase 1: auth, invites, case authoring.
Run with:  python3 -m unittest test_hub_phase1.py
(Skipped automatically when Flask is not installed.)"""

import re
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


@unittest.skipUnless(FLASK, "Flask is not installed")
class HubPhase1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        self.app.config["TESTING"] = True
        self.app.config["SERVER_NAME"] = "hub.test"
        self.client = self.app.test_client()
        db = connect(self.app.config["DATABASE"])
        try:
            pi_id, _ = create_user(db, "PI Person", "pi@example.org", "pi",
                                   invited=False)
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                       (generate_password_hash("pi-password"), pi_id))
            db.commit()
        finally:
            db.close()

    def tearDown(self):
        self.tmp.cleanup()

    # ---- helpers ----

    def login(self, email="pi@example.org", password="pi-password"):
        return self.client.post(
            "/login", data={"email": email, "password": password},
            follow_redirects=True,
        )

    def invite_and_join(self, name="Dr Grader", email="g1@example.org",
                        password="grader-pass"):
        self.login()
        response = self.client.post(
            "/people", data={"name": name, "email": email},
            follow_redirects=True,
        )
        match = re.search(r"/join/([A-Za-z0-9_\-]+)", response.get_data(as_text=True))
        assert match, "invite link not shown"
        token = match.group(1)
        self.client.get("/logout")
        response = self.client.post(
            "/join/{}".format(token),
            data={"name": name, "password": password, "confirm": password},
            follow_redirects=True,
        )
        return response

    # ---- auth ----

    def test_login_wrong_password_rejected(self):
        response = self.login(password="wrong")
        self.assertIn(b"do not match an account", response.data)

    def test_invite_flow_creates_working_grader_account(self):
        response = self.invite_and_join()
        self.assertIn(b"Your account is ready", response.data)
        # The invite link is one-time.
        self.client.get("/logout")
        response = self.login("g1@example.org", "grader-pass")
        self.assertIn(b"My cases", response.data)

    def test_join_link_cannot_be_reused(self):
        self.login()
        response = self.client.post(
            "/people", data={"name": "X", "email": "x@example.org"},
            follow_redirects=True,
        )
        token = re.search(r"/join/([A-Za-z0-9_\-]+)",
                          response.get_data(as_text=True)).group(1)
        self.client.get("/logout")
        self.client.post("/join/{}".format(token),
                         data={"name": "X", "password": "12345678",
                               "confirm": "12345678"})
        response = self.client.get("/join/{}".format(token))
        self.assertEqual(response.status_code, 404)

    def test_people_page_is_pi_only(self):
        self.invite_and_join()
        response = self.client.get("/people")
        self.assertEqual(response.status_code, 403)

    def test_pages_require_login(self):
        response = self.client.get("/cases")
        self.assertEqual(response.status_code, 302)  # to /login

    # ---- case authoring ----

    def test_grader_creates_cases_with_hub_allocated_ids(self):
        self.invite_and_join()
        for text in ("First case.", "Second case."):
            self.client.post("/cases/new", data={
                "case_text": text, "rubric": "item 1\nitem 2",
            })
        response = self.client.get("/cases")
        self.assertIn(b"G001-001", response.data)
        self.assertIn(b"G001-002", response.data)

    def test_case_numbers_never_reused_after_delete(self):
        self.invite_and_join()
        self.client.post("/cases/new",
                         data={"case_text": "One.", "rubric": "r"})
        self.client.post("/cases/G001-001/delete")
        self.client.post("/cases/new",
                         data={"case_text": "Two.", "rubric": "r"})
        response = self.client.get("/cases")
        self.assertNotIn(b'href="/cases/G001-001"', response.data)
        self.assertIn(b'href="/cases/G001-002"', response.data)

    def test_two_graders_get_separate_number_spaces(self):
        self.invite_and_join("A", "a@example.org", "password-a")
        self.client.post("/cases/new", data={"case_text": "A case.",
                                             "rubric": "r"})
        self.client.get("/logout")
        self.invite_and_join("B", "b@example.org", "password-b")
        self.client.post("/cases/new", data={"case_text": "B case.",
                                             "rubric": "r"})
        response = self.client.get("/cases")
        self.assertIn(b"G002-001", response.data)  # B sees only their own
        self.assertNotIn(b"G001-001", response.data)

    def test_rubric_edit_bumps_version_and_keeps_history(self):
        self.invite_and_join()
        self.client.post("/cases/new", data={
            "case_text": "Case.", "rubric": "item 1\nitem 2",
        })
        # Text-only edit: no version bump.
        self.client.post("/cases/G001-001", data={
            "case_text": "Case, clarified.", "rubric": "item 1\nitem 2",
        })
        db = connect(self.app.config["DATABASE"])
        row = db.execute("SELECT * FROM cases WHERE id = 'G001-001'").fetchone()
        self.assertEqual(row["rubric_version"], 1)
        # Rubric edit: bump + history entry.
        self.client.post("/cases/G001-001", data={
            "case_text": "Case, clarified.", "rubric": "item 1 fixed\nitem 2",
        })
        row = db.execute("SELECT * FROM cases WHERE id = 'G001-001'").fetchone()
        db.close()
        self.assertEqual(row["rubric_version"], 2)
        self.assertIn("item 1 fixed", row["rubric_history"])

    def test_graders_cannot_touch_each_others_cases(self):
        self.invite_and_join("A", "a@example.org", "password-a")
        self.client.post("/cases/new", data={"case_text": "A case.",
                                             "rubric": "r"})
        self.client.get("/logout")
        self.invite_and_join("B", "b@example.org", "password-b")
        response = self.client.get("/cases/G001-001")
        self.assertEqual(response.status_code, 404)
        response = self.client.post("/cases/G001-001/delete")
        self.assertEqual(response.status_code, 404)

    def test_pi_sees_all_cases_read_only(self):
        self.invite_and_join()
        self.client.post("/cases/new", data={"case_text": "Case.",
                                             "rubric": "r"})
        self.client.get("/logout")
        self.login()
        response = self.client.get("/cases")
        self.assertIn(b"G001-001", response.data)
        self.assertIn(b"Dr Grader", response.data)
        response = self.client.get("/cases/G001-001")
        self.assertIn(b"only they can", response.data)


if __name__ == "__main__":
    unittest.main()
