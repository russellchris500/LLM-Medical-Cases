"""Tests for the Study Hub phase 2: run jobs, runner API, Hub Runner.
Run with:  python3 -m unittest test_hub_phase2.py
(Skipped automatically when Flask is not installed.)"""

import io
import json
import os
import re
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
    import hub_client
    from hub_client import HubClient, encode_multipart


@unittest.skipUnless(FLASK, "Flask is not installed")
class HubPhase2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        db = connect(self.app.config["DATABASE"])
        try:
            self.pi_id, _ = create_user(db, "PI", "pi@example.org", "pi",
                                        invited=False)
            self.grader_id, _ = create_user(db, "Grader", "g@example.org",
                                            "grader", invited=False)
            for user_id, password in ((self.pi_id, "pi-password"),
                                      (self.grader_id, "grader-pass")):
                db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                           (generate_password_hash(password), user_id))
            # A case owned by the grader.
            db.execute(
                "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
                "VALUES ('G001-001', ?, 1, 'Chest pain case.', ?)",
                (self.grader_id, json.dumps(["ecg", "troponin"])),
            )
            db.execute(
                "UPDATE users SET max_assigned_case_number = 1 WHERE id = ?",
                (self.grader_id,),
            )
            # A runner token for each user.
            self.grader_token = "grader-runner-token"
            self.pi_token = "pi-runner-token"
            db.execute("INSERT INTO runner_tokens (user_id, token_hash) VALUES (?, ?)",
                       (self.grader_id, hash_token(self.grader_token)))
            db.execute("INSERT INTO runner_tokens (user_id, token_hash) VALUES (?, ?)",
                       (self.pi_id, hash_token(self.pi_token)))
            db.commit()
        finally:
            db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def login(self, email="g@example.org", password="grader-pass"):
        return self.client.post("/login", data={"email": email,
                                                "password": password})

    def make_job(self, assignee="me", llm_ids=("testmodel",)):
        self.login()
        data = {"case_id": ["G001-001"], "assignee": assignee,
                "llm_id": list(llm_ids)}
        return self.client.post("/runs", data=data, follow_redirects=True)

    def api(self, token):
        return {"Authorization": "Bearer " + token}

    # ---- job pages ----

    def test_job_creation_and_assignment(self):
        self.make_job("me")
        self.make_job("pi")
        response = self.client.get("/api/runner/jobs",
                                   headers=self.api(self.grader_token))
        jobs = response.get_json()["jobs"]
        self.assertEqual(len(jobs), 1)  # only the self-assigned job
        self.assertEqual(jobs[0]["cases"][0]["case_id"], "G001-001")
        response = self.client.get("/api/runner/jobs",
                                   headers=self.api(self.pi_token))
        self.assertEqual(len(response.get_json()["jobs"]), 1)

    def test_missing_mode_creates_one_job_per_llm_with_unanswered_cases(self):
        db = connect(self.app.config["DATABASE"])
        db.execute(
            "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
            "VALUES ('G001-002', ?, 2, 'Second case.', ?)",
            (self.grader_id, json.dumps(["one item"])),
        )
        db.execute("UPDATE users SET max_assigned_case_number = 2 WHERE id = ?",
                   (self.grader_id,))
        # testmodel already answered the first case; claude answered nothing.
        db.execute(
            "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
            "response_text, status, rubric_version_at_run) "
            "VALUES ('G001-001', ?, 'testmodel', 'testmodel@test-model-1', "
            "'done', 'ok', 1)",
            (self.grader_id,),
        )
        db.commit()
        db.close()
        self.login()
        response = self.client.post("/runs", data={
            "llm_id": ["testmodel", "claude"], "assignee": "me",
            "mode": "missing",
        }, follow_redirects=True)
        self.assertIn(b"Created 2 run job(s)", response.data)
        db = connect(self.app.config["DATABASE"])
        jobs = {
            json.loads(row["llm_ids"])[0]: json.loads(row["case_ids"])
            for row in db.execute("SELECT * FROM run_jobs").fetchall()
        }
        db.close()
        self.assertEqual(jobs["testmodel"], ["G001-002"])
        self.assertEqual(jobs["claude"], ["G001-001", "G001-002"])

    def test_missing_mode_counts_discarded_as_unanswered_and_skips_covered(self):
        db = connect(self.app.config["DATABASE"])
        db.execute(
            "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
            "response_text, status, rubric_version_at_run) "
            "VALUES ('G001-001', ?, 'testmodel', 'testmodel@test-model-1', "
            "'done', 'ok', 1)",
            (self.grader_id,),
        )
        db.commit()
        db.close()
        self.login()
        # Everything answered: no job is created.
        response = self.client.post("/runs", data={
            "llm_id": ["testmodel"], "assignee": "me", "mode": "missing",
        }, follow_redirects=True)
        self.assertIn(b"Nothing to run", response.data)
        db = connect(self.app.config["DATABASE"])
        self.assertEqual(
            db.execute("SELECT COUNT(*) AS n FROM run_jobs").fetchone()["n"], 0
        )
        db.execute("UPDATE answers SET status = 'discarded'")
        db.commit()
        db.close()
        # A discarded answer needs a fresh run, so the case counts again.
        response = self.client.post("/runs", data={
            "llm_id": ["testmodel"], "assignee": "me", "mode": "missing",
        }, follow_redirects=True)
        self.assertIn(b"Created 1 run job(s)", response.data)
        db = connect(self.app.config["DATABASE"])
        row = db.execute("SELECT * FROM run_jobs").fetchone()
        db.close()
        self.assertEqual(json.loads(row["case_ids"]), ["G001-001"])
        self.assertEqual(json.loads(row["llm_ids"]), ["testmodel"])

    # ---- two-level (site, model) selection ----

    def model_id_of(self, llm_id, model_name):
        db = connect(self.app.config["DATABASE"])
        row = db.execute(
            "SELECT id FROM llm_models WHERE llm_id = ? AND model_name = ?",
            (llm_id, model_name),
        ).fetchone()
        db.close()
        return row["id"] if row else None

    def test_model_list_seeds_add_and_hide(self):
        self.login()
        page = self.client.get("/runs")
        self.assertIn(b"test-model-1", page.data)  # seeded
        self.client.post("/runs/models/add", data={
            "llm_id": "gpt", "model_name": "gpt-5-mini",
        })
        page = self.client.get("/runs")
        self.assertIn(b"gpt-5-mini", page.data)
        # Adding the same model twice stays one row.
        self.client.post("/runs/models/add", data={
            "llm_id": "gpt", "model_name": "gpt-5-mini",
        })
        db = connect(self.app.config["DATABASE"])
        count = db.execute(
            "SELECT COUNT(*) AS n FROM llm_models WHERE llm_id = 'gpt' "
            "AND model_name = 'gpt-5-mini'"
        ).fetchone()["n"]
        db.close()
        self.assertEqual(count, 1)
        # Only the PI can hide a model; hiding keeps the row.
        model_id = self.model_id_of("gpt", "gpt-5-mini")
        response = self.client.post("/runs/models/{}/remove".format(model_id))
        self.assertEqual(response.status_code, 403)
        self.client.get("/logout")
        self.login("pi@example.org", "pi-password")
        self.client.post("/runs/models/{}/remove".format(model_id))
        self.client.get("/logout")
        self.login()
        self.assertNotIn(b"gpt-5-mini", self.client.get("/runs").data)

    def test_job_with_two_models_of_one_site(self):
        self.login()
        self.client.post("/runs/models/add", data={
            "llm_id": "gpt", "model_name": "gpt-5-mini",
        })
        ids = [self.model_id_of("gpt", "gpt-5"),
               self.model_id_of("gpt", "gpt-5-mini")]
        self.client.post("/runs", data={
            "case_id": ["G001-001"], "assignee": "me",
            "llm_model": [str(i) for i in ids],
        })
        db = connect(self.app.config["DATABASE"])
        job = db.execute("SELECT * FROM run_jobs").fetchone()
        db.close()
        self.assertEqual(json.loads(job["llm_ids"]), [
            {"llm_id": "gpt", "model_name": "gpt-5"},
            {"llm_id": "gpt", "model_name": "gpt-5-mini"},
        ])
        # The jobs table names both models; the runner payload carries
        # per-model variants plus legacy site ids.
        page = self.client.get("/runs")
        self.assertIn(b"[gpt-5]", page.data)
        self.assertIn(b"[gpt-5-mini]", page.data)
        jobs = self.client.get(
            "/api/runner/jobs", headers=self.api(self.grader_token)
        ).get_json()["jobs"]
        self.assertEqual(jobs[0]["llm_ids"], ["gpt"])
        self.assertEqual(jobs[0]["llms"], [
            {"llm_id": "gpt", "model_name": "gpt-5",
             "variant_id": "gpt@gpt-5"},
            {"llm_id": "gpt", "model_name": "gpt-5-mini",
             "variant_id": "gpt@gpt-5-mini"},
        ])

    def test_missing_mode_is_per_model_not_per_site(self):
        db = connect(self.app.config["DATABASE"])
        # gpt-5 already answered the case; gpt-5-mini never ran.
        db.execute(
            "INSERT INTO answers (case_id, run_by, llm_id, variant_id, "
            "response_text, status, rubric_version_at_run) "
            "VALUES ('G001-001', ?, 'gpt', 'gpt@gpt-5', 'done', 'ok', 1)",
            (self.grader_id,),
        )
        db.commit()
        db.close()
        self.login()
        self.client.post("/runs/models/add", data={
            "llm_id": "gpt", "model_name": "gpt-5-mini",
        })
        ids = [self.model_id_of("gpt", "gpt-5"),
               self.model_id_of("gpt", "gpt-5-mini")]
        response = self.client.post("/runs", data={
            "llm_model": [str(i) for i in ids], "assignee": "me",
            "mode": "missing",
        }, follow_redirects=True)
        self.assertIn(b"Created 1 run job(s)", response.data)
        self.assertIn(b"Already fully answered", response.data)
        db = connect(self.app.config["DATABASE"])
        job = db.execute("SELECT * FROM run_jobs").fetchone()
        db.close()
        self.assertEqual(json.loads(job["llm_ids"]),
                         [{"llm_id": "gpt", "model_name": "gpt-5-mini"}])
        self.assertEqual(json.loads(job["case_ids"]), ["G001-001"])

    def test_job_models_runs_two_variants_and_flags_browser_mismatch(self):
        import hub_runner
        from eval_common import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            settings = SettingsStore.load_or_create(
                os.path.join(tmp, "settings.json")
            )
            settings.browser_model("openevidence")["model"] = "oe-local"
            lines = []
            job = {
                "llm_ids": ["claude", "openevidence"],
                "llms": [
                    {"llm_id": "claude", "model_name": "claude-opus-4-8",
                     "variant_id": "claude@claude-opus-4-8"},
                    {"llm_id": "claude", "model_name": "claude-x",
                     "variant_id": "claude@claude-x"},
                    {"llm_id": "openevidence", "model_name": "oe-job",
                     "variant_id": "openevidence@oe-job"},
                ],
            }
            chosen = hub_runner.job_models(job, settings, lines.append)
            variants = [entry["variant_id"] for entry in chosen]
            self.assertEqual(variants, ["claude@claude-opus-4-8",
                                        "claude@claude-x",
                                        "openevidence@oe-job"])
            names = {entry["variant_id"]: entry["model_name"]
                     for entry in chosen}
            self.assertEqual(names["claude@claude-x"], "claude-x")
            self.assertTrue(any("model picker" in line for line in lines))

    def test_run_api_phase_asks_under_the_entry_model_name(self):
        import run_llms
        from eval_common import AnswersStore, SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            settings = SettingsStore.load_or_create(
                os.path.join(tmp, "settings.json")
            )
            settings.api_model("claude")["api_key"] = "k"
            settings.api_model("claude")["model"] = "from-settings"
            answers = AnswersStore.load_or_create(
                os.path.join(tmp, "answers.json"),
                os.path.join(tmp, "images"),
            )
            master = type("M", (), {"cases": {"G001-001": {
                "case_id": "G001-001", "case_text": "q", "rubric": [],
                "rubric_version": 1,
            }}})()
            model = {
                "model_id": "claude", "kind": "api",
                "model_name": "from-the-job",
                "variant_id": "claude@from-the-job",
                "display_name": "Claude", "scored_as": "Claude (from-the-job)",
            }
            seen = []
            original = run_llms.call_api_model

            def fake_call(model_id, settings_entry, prompt, options, log=print):
                seen.append(settings_entry.get("model"))
                return {"response_text": "ok", "model_requested": "x",
                        "model_reported": "x", "attempts": 1}

            run_llms.call_api_model = fake_call
            try:
                class Ui:
                    stop_requested = False

                    def log(self, message):
                        pass

                run_llms.run_api_phase(master, answers, settings, [model],
                                       {"claude@from-the-job": ["G001-001"]},
                                       Ui())
            finally:
                run_llms.call_api_model = original
            self.assertEqual(seen, ["from-the-job"])

    def test_job_requires_own_cases(self):
        self.login("pi@example.org", "pi-password")
        response = self.client.post("/runs", data={
            "case_id": ["G001-001"], "llm_id": ["testmodel"], "assignee": "me",
        })
        self.assertEqual(response.status_code, 403)

    # ---- runner API ----

    def test_api_requires_valid_token(self):
        response = self.client.get("/api/runner/jobs")
        self.assertEqual(response.status_code, 401)
        response = self.client.get("/api/runner/jobs",
                                   headers=self.api("wrong-token"))
        self.assertEqual(response.status_code, 401)

    def test_answer_upload_stores_files_and_flags_self_identification(self):
        self.make_job("me")
        job_id = 1
        fields = {
            "run_job_id": str(job_id), "case_id": "G001-001",
            "llm_id": "claude", "model_name": "claude-opus-4-8",
            "variant_id": "claude@claude-opus-4-8",
            "model_display_name": "Anthropic Claude (claude-opus-4-8)",
            "response_text": "As Claude, an AI by Anthropic, I would order an ECG.",
            "thinking_setting": "adaptive thinking, effort 'high'",
            "rubric_version": "1", "status": "ok",
        }
        body, content_type = encode_multipart(
            fields,
            [("images", "fig_001.png", b"\x89PNG fake"),
             ("answer_html", "a.html", b"<div>hi</div>")],
        )
        response = self.client.post(
            "/api/runner/answers", data=body,
            content_type=content_type, headers=self.api(self.grader_token),
        )
        payload = response.get_json()
        self.assertTrue(payload["stored"])
        self.assertIn("Claude", payload["self_id_warning"])
        db = connect(self.app.config["DATABASE"])
        row = db.execute("SELECT * FROM answers").fetchone()
        db.close()
        self.assertEqual(row["variant_id"], "claude@claude-opus-4-8")
        self.assertEqual(row["run_by_owner"], 1)  # grader ran their own case
        images = json.loads(row["image_paths"])
        self.assertEqual(len(images), 1)
        self.assertTrue(os.path.exists(
            os.path.join(self.app.config["ANSWER_DIR"], images[0])
        ))
        self.assertTrue(row["answer_html_path"])

    def test_reupload_replaces_not_duplicates(self):
        self.make_job("me")
        for text in ("first", "second"):
            fields = {
                "run_job_id": "1", "case_id": "G001-001",
                "variant_id": "testmodel", "llm_id": "testmodel",
                "response_text": text, "rubric_version": "1",
            }
            body, content_type = encode_multipart(fields, [])
            self.client.post("/api/runner/answers", data=body,
                             content_type=content_type,
                             headers=self.api(self.grader_token))
        db = connect(self.app.config["DATABASE"])
        rows = db.execute("SELECT * FROM answers").fetchall()
        db.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["response_text"], "second")

    def test_foreign_job_and_case_rejected(self):
        self.make_job("me")
        fields = {"run_job_id": "1", "case_id": "G001-001",
                  "variant_id": "testmodel", "llm_id": "testmodel",
                  "response_text": "x", "rubric_version": "1"}
        body, content_type = encode_multipart(fields, [])
        response = self.client.post("/api/runner/answers", data=body,
                                    content_type=content_type,
                                    headers=self.api(self.pi_token))
        self.assertEqual(response.status_code, 404)  # not the PI's job

    def test_job_status_transitions(self):
        self.make_job("me")
        response = self.client.post(
            "/api/runner/jobs/1/status", json={"status": "done", "note": "2 answers"},
            headers=self.api(self.grader_token),
        )
        self.assertTrue(response.get_json()["stored"])
        self.login()
        page = self.client.get("/runs")
        self.assertIn(b'<span class="chip good">done</span>', page.data)
        self.assertIn(b"2 answers", page.data)


@unittest.skipUnless(FLASK, "Flask is not installed")
class HubRunnerIntegrationTests(unittest.TestCase):
    """The real Hub Runner run_job() against a live hub, with the test
    model - the whole phase-2 pipeline end to end."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        self.app = create_app(instance_dir=self.tmp.name, secret_key="test")
        db = connect(self.app.config["DATABASE"])
        self.grader_id, _ = create_user(db, "G", "g@example.org", "grader",
                                        invited=False)
        for number, text in ((1, "Case one."), (2, "Case two.")):
            db.execute(
                "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
                "VALUES (?, ?, ?, ?, ?)",
                ("G001-{:03d}".format(number), self.grader_id, number, text,
                 json.dumps(["r1"])),
            )
        db.execute(
            "INSERT INTO run_jobs (requested_by, assigned_to, case_ids, llm_ids) "
            "VALUES (?, ?, ?, ?)",
            (self.grader_id, self.grader_id,
             json.dumps(["G001-001", "G001-002"]), json.dumps(["testmodel"])),
        )
        self.token = "integration-token"
        db.execute("INSERT INTO runner_tokens (user_id, token_hash) VALUES (?, ?)",
                   (self.grader_id, hash_token(self.token)))
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

    def test_run_job_end_to_end_with_the_test_model(self):
        from eval_common import SettingsStore
        import hub_runner

        settings = SettingsStore.load_or_create("settings.json")
        settings.data["hub"] = {
            "url": "http://127.0.0.1:{}".format(self.port),
            "token": self.token,
        }

        class Ui:
            stop_requested = False

            def __init__(self):
                self.lines = []

            def log(self, message):
                self.lines.append(message)

        ui = Ui()
        client = hub_runner.hub_client_from(settings)
        jobs = client.jobs()
        self.assertEqual(len(jobs), 1)
        ok, failed, uploads_failed = hub_runner.run_job(jobs[0], settings, ui)
        self.assertEqual((ok, failed, uploads_failed), (2, 0, 0))
        hub_runner.finish_job(jobs[0], settings, ok, failed, uploads_failed, ui)

        db = connect(self.app.config["DATABASE"])
        answers = db.execute("SELECT * FROM answers ORDER BY case_id").fetchall()
        job = db.execute("SELECT * FROM run_jobs").fetchone()
        db.close()
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[0]["variant_id"], "testmodel@test-model-1")
        self.assertIn("TEST ANSWER", answers[0]["response_text"])
        self.assertEqual(answers[0]["run_by_owner"], 1)
        self.assertEqual(job["status"], "done")
        # Re-running the same job re-uploads without re-asking.
        jobs = client.jobs()
        self.assertEqual(jobs, [])  # done jobs are no longer offered


if __name__ == "__main__":
    unittest.main()
