"""Tests for the Study Hub phase 3: blinded grading, flags, and instant
rubric invalidation.  Run with:  python3 -m unittest test_hub_phase3.py"""

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
    from hub.cases import rubric_ops
    from hub.db import connect


@unittest.skipUnless(FLASK, "Flask is not installed")
class GradingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        db = connect(self.app.config["DATABASE"])
        self.pi_id, _ = create_user(db, "PI", "pi@example.org", "pi", invited=False)
        self.g1, _ = create_user(db, "Alice", "a@example.org", "grader", invited=False)
        self.g2, _ = create_user(db, "Bob", "b@example.org", "grader", invited=False)
        for user_id, password in ((self.pi_id, "pi-password"),
                                  (self.g1, "alice-pass"), (self.g2, "bob-pass")):
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                       (generate_password_hash(password), user_id))
        db.execute(
            "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
            "VALUES ('G001-001', ?, 1, 'Chest pain.', ?)",
            (self.g1, json.dumps(["orders ECG", "orders troponin",
                                  "gives aspirin"])),
        )
        # Three answers: two variants run by the PI, one variant also run
        # by the owner (the PI's copy must win for blinding).
        answers = [
            ("claude@opus", "PI answer from claude", self.pi_id, 0),
            ("gpt@gpt-5", "PI answer from gpt", self.pi_id, 0),
            ("claude@opus", "owner answer from claude", self.g1, 1),
        ]
        for variant, text, run_by, owner_flag in answers:
            db.execute(
                "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
                "response_text, status, rubric_version_at_run, run_by_owner) "
                "VALUES ('G001-001', ?, 'x', ?, ?, 'ok', 1, ?)",
                (run_by, variant, text, owner_flag),
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

    def open_queue(self):
        self.login()
        return self.client.get("/grade")

    def grade(self, label, items, risk=None, poor=None, comment=""):
        data = {"comment": comment}
        for index, value in enumerate(items):
            data["item_{}".format(index)] = "yes" if value else "no"
        if risk is not None:
            data["risk"] = "yes" if risk else "no"
        if poor is not None:
            data["poor"] = "yes" if poor else "no"
        return self.client.post("/grade/G001-001/{}".format(label), data=data,
                                follow_redirects=True)

    def labels(self):
        self.open_queue()
        db = self.db()
        rows = db.execute(
            "SELECT blind_labels.label, answers.variant_id, answers.run_by_owner "
            "FROM blind_labels JOIN answers ON answers.id = blind_labels.answer_id"
        ).fetchall()
        db.close()
        return {row["variant_id"]: row for row in rows}

    # ---- blinding ----

    def test_one_answer_per_variant_preferring_the_blind_run(self):
        by_variant = self.labels()
        self.assertEqual(len(by_variant), 2)  # one per variant
        self.assertEqual(by_variant["claude@opus"]["run_by_owner"], 0)

    def test_labels_are_stable_and_page_leaks_no_model_names(self):
        first = self.labels()
        second = self.labels()
        self.assertEqual(
            {v: r["label"] for v, r in first.items()},
            {v: r["label"] for v, r in second.items()},
        )
        self.login()
        for label in ("A", "B"):
            page = self.client.get("/grade/G001-001/{}".format(label))
            text = page.get_data(as_text=True)
            for forbidden in ("claude", "gpt", "opus", "variant"):
                self.assertNotIn(forbidden, text.lower().replace(
                    "answer from claude", "").replace("answer from gpt", ""))

    def test_answer_page_keeps_the_sticky_rubric_panel(self):
        # The rubric lives in a side panel that stays on screen while the
        # answer scrolls (and stacks ABOVE the answer on narrow screens).
        self.open_queue()
        page = self.client.get("/grade/G001-001/A")
        text = page.get_data(as_text=True)
        self.assertIn('id="gradepanel"', text)
        self.assertIn("position: sticky", text)
        self.assertLess(text.index('id="gradepanel"'), len(text))

    def test_other_graders_cannot_open_the_case(self):
        self.open_queue()
        self.login("b@example.org", "bob-pass")
        response = self.client.get("/grade/G001-001/A")
        self.assertEqual(response.status_code, 404)

    # ---- grading rules ----

    def test_missed_item_scores_zero_without_risk_question(self):
        self.open_queue()
        response = self.grade("A", [True, False, True])
        self.assertIn(b"scored 0", response.data)

    def test_all_covered_needs_risk_then_poor(self):
        self.open_queue()
        response = self.grade("A", [True, True, True])
        self.assertIn(b"unnecessary-risk question", response.data)
        response = self.grade("A", [True, True, True], risk=False)
        self.assertIn(b"poor-approach question", response.data)
        response = self.grade("A", [True, True, True], risk=False, poor=False)
        self.assertIn(b"scored 2", response.data)
        response = self.grade("B", [True, True, True], risk=True)
        self.assertIn(b"scored 0", response.data)

    def test_regrade_updates_in_place(self):
        self.open_queue()
        self.grade("A", [True, False, True])
        self.grade("A", [True, True, True], risk=False, poor=True)
        db = self.db()
        rows = db.execute("SELECT * FROM grades WHERE superseded = 0").fetchall()
        db.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["score"], 1)

    # ---- flags ----

    def test_flag_reaches_the_case_owner(self):
        self.open_queue()
        self.client.post("/grade/G001-001/flag", data={
            "item_index": "1", "note": "troponin is too strict",
        })
        page = self.client.get("/flags")
        self.assertIn(b"troponin is too strict", page.data)

    # ---- instant rubric invalidation ----

    def edit_rubric(self, new_items):
        return self.client.post("/cases/G001-001", data={
            "case_text": "Chest pain.", "rubric": "\n".join(new_items),
        }, follow_redirects=True)

    def active_grades(self):
        db = self.db()
        rows = db.execute(
            "SELECT grades.*, blind_labels.label FROM grades "
            "JOIN blind_labels ON blind_labels.answer_id = grades.answer_id "
            "AND blind_labels.assignment_id = grades.assignment_id "
            "WHERE grades.superseded = 0 ORDER BY blind_labels.label"
        ).fetchall()
        db.close()
        return rows

    def test_removed_item_keeps_every_other_judgment(self):
        self.open_queue()
        # A misses ONLY troponin (item 1); B misses the ECG item.
        self.grade("A", [True, False, True])
        self.grade("B", [False, True, True])
        response = self.edit_rubric(["orders ECG", "gives aspirin"])
        self.assertIn(b"1 grade(s) carried over in full", response.data)
        self.assertIn(b"1 grade(s) carried over but need finishing",
                      response.data)
        active = self.active_grades()
        self.assertEqual(len(active), 2)
        by_label = {row["label"]: row for row in active}
        # A is now all-covered but its risk/approach questions were never
        # asked - it waits for JUST those (score pending).
        self.assertEqual(json.loads(by_label["A"]["rubric_results"]),
                         [True, True])
        self.assertIsNone(by_label["A"]["score"])
        # B keeps its judgments minus the deleted item.
        self.assertEqual(json.loads(by_label["B"]["rubric_results"]),
                         [False, True])
        self.assertEqual(by_label["B"]["score"], 0)
        for row in active:
            self.assertEqual(row["rubric_version"], 2)

    def test_numbered_rubric_deletion_carries_the_score_2_grade(self):
        # When rubric items carry their own numbers, deleting one
        # renumbers the rest - that must read as a deletion, never as a
        # rewording that touches other judgments.
        self.login()
        self.edit_rubric(["1. orders ECG", "2. orders troponin",
                          "3. gives aspirin"])
        self.open_queue()
        self.grade("A", [True, True, True], risk=False, poor=False)  # a 2
        self.grade("B", [True, False, True])  # missed ONLY the doomed item
        response = self.edit_rubric(["1. orders ECG", "2. gives aspirin"])
        self.assertIn(b"1 grade(s) carried over in full", response.data)
        active = self.active_grades()
        self.assertEqual(len(active), 2)
        by_label = {row["label"]: row for row in active}
        self.assertEqual(by_label["A"]["score"], 2)
        self.assertEqual(json.loads(by_label["A"]["rubric_results"]),
                         [True, True])
        self.assertEqual(by_label["A"]["rubric_version"], 3)
        # B is all-covered now; it owes only the risk/approach answers.
        self.assertIsNone(by_label["B"]["score"])
        self.assertEqual(json.loads(by_label["B"]["rubric_results"]),
                         [True, True])

    def test_plain_deletion_carries_a_score_2_grade(self):
        self.open_queue()
        self.grade("A", [True, True, True], risk=False, poor=False)
        self.edit_rubric(["orders ECG", "gives aspirin"])
        db = self.db()
        active = db.execute(
            "SELECT * FROM grades WHERE superseded = 0").fetchall()
        db.close()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["score"], 2)
        self.assertEqual(json.loads(active[0]["rubric_results"]),
                         [True, True])

    def test_pure_reorder_carries_grades_with_results_remapped(self):
        self.open_queue()
        self.grade("A", [True, False, True])  # missed troponin
        self.edit_rubric(["gives aspirin", "orders ECG", "orders troponin"])
        db = self.db()
        active = db.execute(
            "SELECT * FROM grades WHERE superseded = 0").fetchall()
        db.close()
        self.assertEqual(len(active), 1)
        # Results follow their items to the new positions.
        self.assertEqual(json.loads(active[0]["rubric_results"]),
                         [True, True, False])
        self.assertEqual(active[0]["score"], 0)

    def test_rewording_asks_then_reset_blanks_only_that_item(self):
        self.open_queue()
        self.grade("A", [True, True, True], risk=False, poor=False)
        # First save: the hub asks what to do with judgments on the
        # changed line - nothing is applied yet.
        response = self.edit_rubric(["orders ECG", "orders troponin and CK",
                                     "gives aspirin"])
        self.assertIn(b"was changed", response.data)
        db = self.db()
        version = db.execute(
            "SELECT rubric_version FROM cases").fetchone()["rubric_version"]
        db.close()
        self.assertEqual(version, 1)
        # Confirm: reset judgments for that line only.
        response = self.client.post("/cases/G001-001", data={
            "case_text": "Chest pain.",
            "rubric": "orders ECG\norders troponin and CK\ngives aspirin",
            "changed_1": "reset",
        }, follow_redirects=True)
        self.assertIn(b"need finishing", response.data)
        active = self.active_grades()
        self.assertEqual(len(active), 1)
        self.assertEqual(json.loads(active[0]["rubric_results"]),
                         [True, None, True])
        self.assertIsNone(active[0]["score"])
        self.assertEqual(active[0]["rubric_version"], 2)

    def test_added_item_leaves_other_judgments_in_place(self):
        self.open_queue()
        self.grade("A", [True, True, True], risk=False, poor=False)
        self.grade("B", [True, True, True], risk=False, poor=False)
        response = self.edit_rubric(["orders ECG", "orders troponin",
                                     "gives aspirin", "checks renal function"])
        # No question needed for a pure addition - grades carry, pending
        # only the new item.
        self.assertIn(b"2 grade(s) carried over but need finishing",
                      response.data)
        active = self.active_grades()
        self.assertEqual(len(active), 2)
        for row in active:
            self.assertEqual(json.loads(row["rubric_results"]),
                             [True, True, True, None])
            self.assertIsNone(row["score"])
            self.assertEqual(row["rubric_version"], 2)

    def test_rubric_ops_diff(self):
        ops = rubric_ops(["a", "b", "c"], ["a", "c"])
        self.assertEqual(ops, {"removed": [1], "reworded": [], "added": []})
        ops = rubric_ops(["a", "b"], ["a", "b reworded"])
        self.assertEqual(ops, {"removed": [], "reworded": [1], "added": []})
        ops = rubric_ops(["a"], ["a", "new"])
        self.assertEqual(ops, {"removed": [], "reworded": [], "added": [1]})

    # ---- cross-grading ----

    def test_pi_assigns_cross_grading_and_bob_grades_blind(self):
        self.login("pi@example.org", "pi-password")
        self.client.post("/assignments", data={
            "grader_id": str(self.g2), "case_id": ["G001-001"],
        })
        self.login("b@example.org", "bob-pass")
        page = self.client.get("/grade")
        self.assertIn(b"cross-grading", page.data)
        response = self.grade("A", [True, True, True], risk=False, poor=False)
        self.assertIn(b"scored 2", response.data)
        db = self.db()
        graders = db.execute(
            "SELECT DISTINCT grading_assignments.grader_id FROM grades "
            "JOIN grading_assignments "
            "ON grading_assignments.id = grades.assignment_id"
        ).fetchall()
        db.close()
        self.assertEqual({row["grader_id"] for row in graders}, {self.g2})


if __name__ == "__main__":
    unittest.main()
