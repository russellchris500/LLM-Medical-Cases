"""Tests: LLM-as-a-judge runs, verification against human grades,
self-judging policy, rubric edit + retest, and the Runner end to end.
Run with:  python3 -m unittest test_hub_judge.py"""

import json
import os
import tempfile
import threading
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
class JudgeTests(unittest.TestCase):
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
            (self.g1, json.dumps(["orders ECG", "gives aspirin"])),
        )
        db.execute("UPDATE users SET max_assigned_case_number = 1 WHERE id = ?",
                   (self.g1,))
        for llm_id, variant, text in (("claude", "claude@claude-x", "ECG + aspirin"),
                                      ("gpt", "gpt@gpt-5", "ECG only")):
            db.execute(
                "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
                "model_display_name, response_text, status, "
                "rubric_version_at_run, run_by_owner) "
                "VALUES ('G001-001', ?, ?, ?, ?, ?, 'ok', 1, 0)",
                (self.pi_id, llm_id, variant, variant, text),
            )
        self.token = "alice-runner"
        db.execute("INSERT INTO runner_tokens (user_id, token_hash) VALUES (?, ?)",
                   (self.g1, hash_token(self.token)))
        db.commit()
        db.close()
        # Alice grades both answers: claude -> 2, gpt -> 0 (missed aspirin).
        self.login()
        self.client.get("/runs")  # seeds the model list
        self.client.post("/runs/models/add", data={
            "llm_id": "claude", "model_name": "claude-x",
        })
        self.client.get("/grade")
        for variant, items in (("claude@claude-x", ["yes", "yes"]),
                               ("gpt@gpt-5", ["yes", "no"])):
            label = self.label_of(variant)
            data = {"item_0": items[0], "item_1": items[1]}
            if items == ["yes", "yes"]:
                data.update(risk="no", poor="no")
            self.client.post("/grade/G001-001/{}".format(label), data=data)

    def tearDown(self):
        self.tmp.cleanup()

    def db(self):
        return connect(self.app.config["DATABASE"])

    def login(self, email="a@example.org", password="alice-pass"):
        self.client.get("/logout")
        return self.client.post("/login", data={"email": email,
                                                "password": password})

    def label_of(self, variant):
        db = self.db()
        row = db.execute(
            "SELECT blind_labels.label FROM blind_labels "
            "JOIN answers ON answers.id = blind_labels.answer_id "
            "WHERE answers.variant_id = ?", (variant,),
        ).fetchone()
        db.close()
        return row["label"]

    def answer_id(self, variant):
        db = self.db()
        row = db.execute("SELECT id FROM answers WHERE variant_id = ?",
                         (variant,)).fetchone()
        db.close()
        return row["id"]

    def judge_model_id(self):
        db = self.db()
        row = db.execute("SELECT id FROM llm_models WHERE llm_id = 'claude' "
                         "AND model_name = 'claude-x'").fetchone()
        db.close()
        return row["id"]

    def create_run(self, allow_self=False):
        data = {"judge_model": str(self.judge_model_id()),
                "case_id": ["G001-001"], "assignee": "me"}
        if allow_self:
            data["allow_self"] = "1"
        return self.client.post("/judge", data=data, follow_redirects=True)

    def api(self):
        return {"Authorization": "Bearer " + self.token}

    def runner_runs(self):
        return self.client.get("/api/runner/judge-runs",
                               headers=self.api()).get_json()["judge_runs"]

    def upload(self, run_id, variant, results, risk=False, poor=False):
        return self.client.post("/api/runner/judge-grades", json={
            "judge_run_id": run_id, "answer_id": self.answer_id(variant),
            "rubric_version": 1, "status": "ok", "results": results,
            "evidence": ["e1", "e2"], "unnecessary_risk": risk,
            "poor_approach": poor, "raw_response": "{}",
        }, headers=self.api())

    # ---- self-judging policy ----

    def test_run_excludes_own_answers_unless_allowed(self):
        response = self.create_run()
        self.assertIn(b"Judge run #1 created", response.data)
        runs = self.runner_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["judge_variant"], "claude@claude-x")
        variants = [self.variant_for(a["answer_id"]) for a in runs[0]["answers"]]
        self.assertEqual(variants, ["gpt@gpt-5"])  # its own answer left out
        self.assertFalse(runs[0]["answers"][0]["self_judged"])
        self.create_run(allow_self=True)
        runs = self.runner_runs()
        flags = {self.variant_for(a["answer_id"]): a["self_judged"]
                 for a in runs[1]["answers"]}
        self.assertEqual(flags, {"claude@claude-x": True, "gpt@gpt-5": False})

    def variant_for(self, answer_id):
        db = self.db()
        row = db.execute("SELECT variant_id FROM answers WHERE id = ?",
                         (answer_id,)).fetchone()
        db.close()
        return row["variant_id"]

    def test_only_api_models_can_judge(self):
        self.client.post("/runs/models/add", data={
            "llm_id": "openevidence", "model_name": "oe-2026",
        })
        page = self.client.get("/judge")
        self.assertIn(b"- claude-x</option>", page.data)
        self.assertNotIn(b"- oe-2026</option>", page.data)

    # ---- verification against human grades ----

    def test_accepted_only_when_every_item_matches(self):
        self.create_run(allow_self=True)
        # Judge agrees with Alice on gpt (missed aspirin) ...
        self.assertTrue(self.upload(1, "gpt@gpt-5", [True, False])
                        .get_json()["stored"])
        # ... but disagrees on claude (says aspirin missed; Alice: covered).
        self.upload(1, "claude@claude-x", [True, False])
        page = self.client.get("/judge").get_data(as_text=True)
        self.assertIn("1/2 (50%)", page)
        db = self.db()
        rows = {row["self_judged"]: row for row in
                db.execute("SELECT * FROM judge_grades").fetchall()}
        db.close()
        self.assertEqual(rows[0]["score"], 0)   # gpt, not self
        self.assertEqual(rows[1]["score"], 0)   # claude, self-judged
        case_page = self.client.get("/judge/G001-001").get_data(as_text=True)
        self.assertIn("accepted", case_page)
        self.assertIn("not accepted", case_page)
        self.assertIn("own answer", case_page)
        self.assertIn('class="miss"', case_page)
        csv_text = self.client.get("/judge.csv").get_data(as_text=True)
        self.assertIn("gpt@gpt-5,claude@claude-x,0,0,0,Alice,1,,0,0,CM,CM", csv_text)
        self.assertIn("claude@claude-x,claude@claude-x,1,0,0,Alice,0,2,2,0,CM,CC",
                      csv_text)

    def test_rubric_edit_marks_stale_and_retest_uses_new_version(self):
        self.create_run(allow_self=True)
        self.upload(1, "claude@claude-x", [True, True])
        self.upload(1, "gpt@gpt-5", [True, False])
        # Delete the aspirin item: the claude grade carries over complete,
        # the gpt grade (it missed ONLY that item) waits for its risk/
        # approach answers; both judge grades are now stale.
        self.client.post("/cases/G001-001", data={
            "case_text": "Chest pain.", "rubric": "orders ECG",
        }, follow_redirects=True)
        page = self.client.get("/judge").get_data(as_text=True)
        self.assertIn("2 - retest", page)
        case_page = self.client.get("/judge/G001-001").get_data(as_text=True)
        self.assertEqual(case_page.count("stale - retest"), 2)
        self.assertIn("No complete human", case_page)  # gpt, pending
        self.assertIn("Retest judge claude-x", case_page)
        response = self.client.post("/judge/G001-001/retest", data={
            "judge_llm_id": "claude", "judge_model_name": "claude-x",
            "allow_self": "1",
        }, follow_redirects=True)
        self.assertIn(b"rubric version 2", response.data)
        db = self.db()
        run = db.execute("SELECT * FROM judge_runs ORDER BY id DESC").fetchone()
        db.close()
        self.assertEqual(json.loads(run["rubric_versions"]), {"G001-001": 2})
        self.assertEqual(run["allow_self"], 1)
        # The new run hands the runner the ONE-item rubric.
        runs = self.runner_runs()
        newest = [r for r in runs if r["id"] == run["id"]][0]
        self.assertEqual(newest["answers"][0]["rubric"], ["orders ECG"])
        self.assertEqual(newest["answers"][0]["rubric_version"], 2)

    def test_home_counts_waiting_judge_runs(self):
        self.create_run()
        page = self.client.get("/home")
        self.assertIn(b"AI judge run", page.data)


@unittest.skipUnless(FLASK, "Flask is not installed")
class JudgeRunnerIntegrationTests(unittest.TestCase):
    """The real Run AI Answers judge loop against a live hub with the
    built-in test model as judge."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        db = connect(self.app.config["DATABASE"])
        self.g1, _ = create_user(db, "G", "g@example.org", "grader", invited=False)
        db.execute(
            "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
            "VALUES ('G001-001', ?, 1, 'Case.', ?)",
            (self.g1, json.dumps(["r1", "r2"])),
        )
        for variant in ("claude@c1", "gpt@g1"):
            db.execute(
                "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
                "response_text, status, rubric_version_at_run) "
                "VALUES ('G001-001', ?, ?, ?, 'answer', 'ok', 1)",
                (self.g1, variant.split("@")[0], variant),
            )
        db.execute("INSERT INTO grading_assignments (grader_id, case_id) "
                   "VALUES (?, 'G001-001')", (self.g1,))
        for answer_id, results in ((1, "[true, true]"), (2, "[true, false]")):
            db.execute(
                "INSERT INTO grades (assignment_id, answer_id, rubric_results, "
                "unnecessary_risk, poor_approach, score, rubric_version) "
                "VALUES (1, ?, ?, ?, ?, ?, 1)",
                (answer_id, results, 0 if results == "[true, true]" else None,
                 0 if results == "[true, true]" else None,
                 2 if results == "[true, true]" else 0),
            )
        db.execute(
            "INSERT INTO judge_runs (requested_by, assigned_to, judge_llm_id, "
            "judge_model_name, judge_variant, case_ids) "
            "VALUES (?, ?, 'testmodel', 'test-model-1', 'testmodel@test-model-1', "
            "'[\"G001-001\"]')",
            (self.g1, self.g1),
        )
        self.token = "judge-token"
        db.execute("INSERT INTO runner_tokens (user_id, token_hash) VALUES (?, ?)",
                   (self.g1, hash_token(self.token)))
        db.commit()
        db.close()
        from werkzeug.serving import make_server

        self.server = make_server("127.0.0.1", 0, self.app)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        os.chdir(self.workdir.name)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.workdir.cleanup()
        self.tmp.cleanup()

    def test_judge_run_end_to_end(self):
        from eval_common import SettingsStore
        import hub_runner

        settings = SettingsStore.load_or_create("settings.json")
        settings.data["hub"] = {
            "url": "http://127.0.0.1:{}".format(self.port), "token": self.token,
        }

        class Ui:
            stop_requested = False

            def __init__(self):
                self.lines = []

            def log(self, message):
                self.lines.append(message)

        ui = Ui()
        client = hub_runner.hub_client_from(settings)
        runs = client.judge_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]["answers"]), 2)
        ok, failed, uploads_failed = hub_runner.run_judge_run(runs[0], settings, ui)
        self.assertEqual((ok, failed, uploads_failed), (2, 0, 0))
        hub_runner.finish_judge_run(runs[0], settings, ok, failed,
                                    uploads_failed, ui)
        db = connect(self.app.config["DATABASE"])
        grades = db.execute("SELECT * FROM judge_grades ORDER BY answer_id").fetchall()
        run = db.execute("SELECT * FROM judge_runs").fetchone()
        db.close()
        self.assertEqual(run["status"], "done")
        self.assertEqual([g["score"] for g in grades], [2, 2])  # test judge says all covered
        self.assertEqual(grades[0]["judge_variant"], "testmodel@test-model-1")
        # Running again skips everything already judged.
        runs = client.judge_runs()
        self.assertEqual(runs, [])  # the run is done, nothing waiting
        self.assertTrue(any("judged, score" in line for line in ui.lines))


if __name__ == "__main__":
    unittest.main()
