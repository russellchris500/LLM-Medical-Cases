"""Tests for the Study Hub phase 4: rankings dashboard and legacy import.
Run with:  python3 -m unittest test_hub_phase4.py"""

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
    from hub.db import connect
    from hub.manage import import_legacy


@unittest.skipUnless(FLASK, "Flask is not installed")
class RankingsTests(unittest.TestCase):
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
        # Three cases; strong model scores 2 everywhere, weak scores 0.
        for number in (1, 2, 3):
            case_id = "G001-{:03d}".format(number)
            db.execute(
                "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
                "VALUES (?, ?, ?, 'Case.', ?)",
                (case_id, self.g1, number, json.dumps(["r1"])),
            )
            for variant, display in (("strong@m1", "Strong (m1)"),
                                     ("weak@m2", "Weak (m2)")):
                db.execute(
                    "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
                    "model_display_name, response_text, status, "
                    "rubric_version_at_run) VALUES (?, ?, 'x', ?, ?, 'a', 'ok', 1)",
                    (case_id, self.pi_id, variant, display),
                )
            db.execute(
                "INSERT INTO grading_assignments (grader_id, case_id, kind) "
                "VALUES (?, ?, 'own')",
                (self.g1, case_id),
            )
        assignments = {row["case_id"]: row["id"] for row in db.execute(
            "SELECT * FROM grading_assignments"
        )}
        for row in db.execute("SELECT * FROM answers").fetchall():
            score = 2 if row["variant_id"].startswith("strong") else 0
            db.execute(
                "INSERT INTO grades (assignment_id, answer_id, rubric_results, "
                "score, rubric_version) VALUES (?, ?, ?, ?, 1)",
                (assignments[row["case_id"]], row["id"],
                 json.dumps([True] if score == 2 else [False]), score),
            )
        db.commit()
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def login(self, email="pi@example.org", password="pi-password"):
        self.client.get("/logout")
        return self.client.post("/login", data={"email": email,
                                                "password": password})

    def test_rankings_order_and_access(self):
        self.login()
        page = self.client.get("/rankings")
        text = page.get_data(as_text=True)
        self.assertLess(text.index("Strong (m1)"), text.index("Weak (m2)"))
        self.assertIn("6 graded answers", text)
        # Graders cannot open the global ranking.
        self.login("a@example.org", "alice-pass")
        self.assertEqual(self.client.get("/rankings").status_code, 403)

    def test_stale_rubric_version_grades_are_excluded(self):
        db = connect(self.app.config["DATABASE"])
        db.execute("UPDATE cases SET rubric_version = 2 WHERE id = 'G001-001'")
        db.commit()
        db.close()
        self.login()
        text = self.client.get("/rankings").get_data(as_text=True)
        self.assertIn("4 graded answers", text)  # 2 of 6 excluded
        self.assertIn("awaiting re-grade", text)

    def test_csv_and_snapshot(self):
        self.login()
        csv_data = self.client.get("/rankings.csv").get_data(as_text=True)
        self.assertIn("Strong (m1)", csv_data.splitlines()[1])
        self.client.post("/rankings/snapshot", follow_redirects=True)
        db = connect(self.app.config["DATABASE"])
        count = db.execute(
            "SELECT COUNT(*) AS n FROM ranking_snapshots"
        ).fetchone()["n"]
        db.close()
        self.assertEqual(count, 1)

    def test_my_results_shows_only_own_data(self):
        self.login("a@example.org", "alice-pass")
        text = self.client.get("/my-results").get_data(as_text=True)
        self.assertIn("Strong (m1)", text)
        self.assertIn("2.0", text)


@unittest.skipUnless(FLASK, "Flask is not installed")
class LegacyImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.legacy = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.legacy.name)
        # Build a real legacy study with the desktop modules.
        from case_editor import CaseStore
        from merge_cases import MasterStore
        from eval_common import AnswersStore, case_hash
        from build_scoring_package import build_package
        from score_answers import Package, ScoresStore, scores_path_for

        provider = CaseStore(3, "provider_003_cases.json")
        provider.add_case("Legacy case one.", ["item 1", "item 2"])
        provider.add_case("Legacy case two.", ["only item"])
        master = MasterStore("master_cases.json")
        master.merge_provider(provider)
        master.save()
        answers = AnswersStore.load_or_create("answers.json", "answer_images")
        os.makedirs("answer_images", exist_ok=True)
        image = os.path.join("answer_images", "003-001_ma_001.png")
        with open(image, "wb") as f:
            f.write(b"\x89PNG legacy image")
        for case_id in ("003-001", "003-002"):
            answers.upsert({
                "case_id": case_id, "model_id": "modela@v1",
                "llm_id": "modela", "model_name": "v1",
                "model_display_name": "Model A (v1)", "status": "ok",
                "response_text": "Legacy answer for " + case_id,
                "images": [image] if case_id == "003-001" else [],
                "case_sha256": case_hash(master.cases[case_id]),
            })
        zip_path, key_path = build_package(
            "legacy", ["003-001", "003-002"], ["modela@v1"], master, answers,
            warn=lambda *_: None,
        )
        package = Package.load(zip_path)
        scores = ScoresStore.load_or_create(
            scores_path_for(os.path.basename(zip_path)), package.manifest
        )
        scores.scorer = "CR"
        with open(key_path, encoding="utf-8") as f:
            key = json.load(f)["key"]
        for case_id in ("003-001", "003-002"):
            label = next(iter(key[case_id]))
            results = [True, True] if case_id == "003-001" else [False]
            risk = False if all(results) else None
            poor = False if all(results) else None
            scores.upsert(case_id, label, results, risk, poor, rubric_version=1)

        os.chdir(self.old_cwd)
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        db = connect(self.app.config["DATABASE"])
        create_user(db, "PI", "pi@example.org", "pi", invited=False)
        self.grader_id, _ = create_user(db, "Alice", "a@example.org", "grader",
                                        invited=False)
        db.commit()
        db.close()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.legacy.cleanup()
        self.tmp.cleanup()

    def test_import_brings_cases_answers_images_and_grades(self):
        report = import_legacy(self.app, "a@example.org", self.legacy.name)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["answers"], 2)
        self.assertEqual(report["grades"], 2)
        self.assertEqual(report["skipped"], [])
        db = connect(self.app.config["DATABASE"])
        case = db.execute("SELECT * FROM cases WHERE id = '003-001'").fetchone()
        self.assertEqual(case["owner_id"], self.grader_id)
        answer = db.execute(
            "SELECT * FROM answers WHERE case_id = '003-001'"
        ).fetchone()
        self.assertEqual(answer["variant_id"], "modela@v1")
        images = json.loads(answer["image_paths"])
        self.assertEqual(len(images), 1)
        self.assertTrue(os.path.exists(
            os.path.join(self.app.config["ANSWER_DIR"], images[0])
        ))
        grades = db.execute("SELECT * FROM grades ORDER BY id").fetchall()
        db.close()
        self.assertEqual([g["score"] for g in grades], [2, 0])
        # Idempotent: importing again adds nothing.
        report = import_legacy(self.app, "a@example.org", self.legacy.name)
        db = connect(self.app.config["DATABASE"])
        count = db.execute("SELECT COUNT(*) AS n FROM grades").fetchone()["n"]
        db.close()
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
