"""Tests: robust case editing - the answers question on a case-text
change, surgical grade carry-over for added/deleted/changed rubric
lines, pending grades, and the builder<->grader shortcut buttons.
Run with:  python3 -m unittest test_hub_case_edit.py"""

import json
import os
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
    from hub.db import MIGRATIONS, connect, init_db


@unittest.skipUnless(FLASK, "Flask is not installed")
class CaseEditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        db = connect(self.app.config["DATABASE"])
        self.pi_id, _ = create_user(db, "PI", "pi@example.org", "pi",
                                    invited=False)
        self.g1, _ = create_user(db, "Alice", "a@example.org", "grader",
                                 invited=False)
        for user_id, password in ((self.pi_id, "pi-password"),
                                  (self.g1, "alice-pass")):
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                       (generate_password_hash(password), user_id))
        db.execute(
            "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
            "VALUES ('G001-001', ?, 1, 'Chest pain.', ?)",
            (self.g1, json.dumps(["orders ECG", "orders troponin"])),
        )
        db.execute(
            "UPDATE users SET max_assigned_case_number = 1 WHERE id = ?",
            (self.g1,),
        )
        for variant, text in (("claude@opus", "answer one"),
                              ("gpt@gpt-5", "answer two")):
            db.execute(
                "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
                "model_display_name, response_text, status, "
                "rubric_version_at_run, run_by_owner) "
                "VALUES ('G001-001', ?, 'x', ?, ?, ?, 'ok', 1, 1)",
                (self.g1, variant, variant, text),
            )
        db.commit()
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def db(self):
        return connect(self.app.config["DATABASE"])

    def login(self, email="a@example.org", password="alice-pass"):
        self.client.get("/logout")
        return self.client.post("/login", data={"email": email,
                                                "password": password})

    def grade_all(self):
        """Alice grades both answers: A and B both score 2."""
        self.login()
        self.client.get("/grade")
        for label in ("A", "B"):
            self.client.post("/grade/G001-001/{}".format(label), data={
                "item_0": "yes", "item_1": "yes", "risk": "no", "poor": "no",
            })

    def save_case(self, case_text, rubric_lines, extra=None,
                  follow_redirects=True):
        data = {"case_text": case_text, "rubric": "\n".join(rubric_lines)}
        data.update(extra or {})
        return self.client.post("/cases/G001-001", data=data,
                                follow_redirects=follow_redirects)

    def active_grades(self):
        db = self.db()
        rows = db.execute("SELECT * FROM grades WHERE superseded = 0").fetchall()
        db.close()
        return rows

    # ---- case-text edits and the answers question ----

    def test_text_edit_with_answers_asks_and_keep_changes_nothing(self):
        self.grade_all()
        response = self.save_case("Chest pain, new wording.",
                                  ["orders ECG", "orders troponin"])
        self.assertIn(b"answer(s)\n            already exist"
                      .replace(b"\n            ", b" "),
                      response.data.replace(b"\n            ", b" "))
        # Nothing applied yet.
        db = self.db()
        text = db.execute("SELECT case_text FROM cases").fetchone()["case_text"]
        db.close()
        self.assertEqual(text, "Chest pain.")
        response = self.save_case("Chest pain, new wording.",
                                  ["orders ECG", "orders troponin"],
                                  {"answers_action": "keep"})
        self.assertIn(b"Saved case G001-001", response.data)
        db = self.db()
        text = db.execute("SELECT case_text FROM cases").fetchone()["case_text"]
        ok = db.execute("SELECT COUNT(*) AS n FROM answers "
                        "WHERE status = 'ok'").fetchone()["n"]
        db.close()
        self.assertEqual(text, "Chest pain, new wording.")
        self.assertEqual(ok, 2)
        self.assertEqual(len(self.active_grades()), 2)

    def test_text_edit_delete_discards_answers_for_rerun(self):
        self.grade_all()
        response = self.save_case("A different question entirely.",
                                  ["orders ECG", "orders troponin"],
                                  {"answers_action": "delete"})
        self.assertIn(b"2 answer(s) were discarded", response.data)
        db = self.db()
        statuses = [row["status"] for row in
                    db.execute("SELECT status FROM answers").fetchall()]
        reasons = {row["discarded_reason"] for row in
                   db.execute("SELECT discarded_reason FROM answers").fetchall()}
        db.close()
        self.assertEqual(statuses, ["discarded", "discarded"])
        self.assertEqual(reasons, {"case text edited"})
        self.assertEqual(len(self.active_grades()), 0)
        page = self.client.get("/runs")
        self.assertIn(b"needs a re-run", page.data)
        self.assertIn(b"case text edited", page.data)

    def test_text_edit_without_answers_saves_directly(self):
        self.login()
        self.client.post("/cases/new", data={
            "case_text": "Fresh case.", "rubric": "one item",
        })
        response = self.client.post("/cases/G001-002", data={
            "case_text": "Fresh case, edited.", "rubric": "one item",
        }, follow_redirects=True)
        self.assertIn(b"Saved case G001-002", response.data)
        db = self.db()
        text = db.execute("SELECT case_text FROM cases WHERE id = 'G001-002'"
                          ).fetchone()["case_text"]
        db.close()
        self.assertEqual(text, "Fresh case, edited.")

    # ---- added lines: grade only the new item ----

    def test_added_line_pends_only_the_new_item_then_completes(self):
        self.grade_all()
        self.save_case("Chest pain.",
                       ["orders ECG", "orders troponin", "gives aspirin"])
        for row in self.active_grades():
            self.assertEqual(json.loads(row["rubric_results"]),
                             [True, True, None])
            self.assertIsNone(row["score"])
        # Pending grades stay out of the results pages.
        self.login("pi@example.org", "pi-password")
        page = self.client.get("/rankings")
        self.assertIn(b"No usable grades yet", page.data)
        self.assertIn(b"awaiting completion", page.data)
        self.login()
        self.assertNotIn(b"claude", self.client.get("/my-results").data)
        # The form comes back prefilled except the new item.
        page = self.client.get("/grade/G001-001/A").get_data(as_text=True)
        new_item_part = page[page.index('name="item_2"'):page.index("riskrow")]
        self.assertNotIn("checked", new_item_part)
        risk_part = page[page.index("riskrow"):]
        self.assertIn("checked", risk_part)  # risk/poor answers carried
        # Completing the grade restores a full score.
        response = self.client.post("/grade/G001-001/A", data={
            "item_0": "yes", "item_1": "yes", "item_2": "yes",
            "risk": "no", "poor": "no",
        }, follow_redirects=True)
        self.assertIn(b"scored 2", response.data)
        scores = sorted((row["score"] is None for row in self.active_grades()))
        self.assertEqual(scores, [False, True])  # one done, one pending

    # ---- changed lines: keep or reset, others untouched ----

    def test_changed_line_keep_carries_the_old_judgment(self):
        self.grade_all()
        response = self.save_case("Chest pain.",
                                  ["orders ECG", "orders troponin and CK"])
        self.assertIn(b"was changed", response.data)
        response = self.save_case("Chest pain.",
                                  ["orders ECG", "orders troponin and CK"],
                                  {"changed_1": "keep"})
        self.assertIn(b"2 grade(s) carried over in full", response.data)
        for row in self.active_grades():
            self.assertEqual(json.loads(row["rubric_results"]), [True, True])
            self.assertEqual(row["score"], 2)
            self.assertEqual(row["rubric_version"], 2)

    def test_changed_line_reset_blanks_only_that_item(self):
        self.grade_all()
        self.save_case("Chest pain.",
                       ["orders ECG", "orders troponin and CK"],
                       {"changed_1": "reset"})
        for row in self.active_grades():
            self.assertEqual(json.loads(row["rubric_results"]), [True, None])
            self.assertIsNone(row["score"])

    def test_changed_line_without_grades_never_asks(self):
        self.login()
        response = self.save_case("Chest pain.",
                                  ["orders ECG", "orders troponin and CK"])
        self.assertIn(b"Saved case G001-001", response.data)

    # ---- shortcut buttons ----

    def test_builder_links_to_grading_and_back(self):
        self.login()
        # Builder -> grading (works even before the queue page was opened,
        # thanks to the assignment fallback in case_page).
        page = self.client.get("/cases/G001-001")
        self.assertIn(b'href="/grade/G001-001"', page.data)
        page = self.client.get("/grade/G001-001")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'href="/cases/G001-001"', page.data)
        self.assertIn(b"Edit this case (builder)", page.data)


@unittest.skipUnless(FLASK, "Flask is not installed")
class MigrationTests(unittest.TestCase):
    def test_0006_rebuild_preserves_grades_and_allows_null_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.db")
            db = connect(path)
            db.execute(
                "CREATE TABLE schema_migrations (version TEXT PRIMARY KEY, "
                "applied_at TEXT NOT NULL DEFAULT "
                "(strftime('%Y-%m-%dT%H:%M:%SZ','now')))"
            )
            for version, statements in MIGRATIONS:
                if version >= "0006":
                    break
                db.executescript(statements)
                db.execute("INSERT INTO schema_migrations (version) "
                           "VALUES (?)", (version,))
            db.execute("INSERT INTO users (name, email, role) "
                       "VALUES ('A', 'a@x.org', 'grader')")
            db.execute("INSERT INTO cases (id, owner_id, case_number, "
                       "case_text, rubric) VALUES ('G001-001', 1, 1, 'c', '[]')")
            db.execute("INSERT INTO grading_assignments (grader_id, case_id) "
                       "VALUES (1, 'G001-001')")
            db.execute("INSERT INTO answers (case_id, run_by, llm_id, "
                       "variant_id) VALUES ('G001-001', 1, 'x', 'v')")
            db.execute("INSERT INTO grades (assignment_id, answer_id, "
                       "rubric_results, score, rubric_version) "
                       "VALUES (1, 1, '[true]', 2, 1)")
            db.commit()
            # The old schema rejects a pending (NULL) score.
            import sqlite3
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE grades SET score = NULL WHERE id = 1")
            db.close()
            init_db(path)
            db = connect(path)
            row = db.execute("SELECT * FROM grades").fetchone()
            self.assertEqual(row["score"], 2)
            self.assertEqual(row["rubric_results"], "[true]")
            db.execute("UPDATE grades SET score = NULL WHERE id = 1")
            db.commit()
            self.assertIsNone(db.execute(
                "SELECT score FROM grades").fetchone()["score"])
            db.close()


if __name__ == "__main__":
    unittest.main()
