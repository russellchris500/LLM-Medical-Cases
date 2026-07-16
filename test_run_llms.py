"""Tests for the (LLM, model name) scoring identity in run_llms.py.
Run with:  python3 -m unittest test_run_llms.py"""

import os
import tempfile
import threading
import unittest

import run_llms
from case_editor import CaseStore
from merge_cases import MasterStore
from eval_common import AnswersStore, SettingsStore, model_slug
from llm_api import ModelAbort
from run_llms import (
    answer_needs_reask,
    build_worklist,
    model_catalog,
    new_record,
    run_api_phase,
)
from rank_llms import build_matches, display_map


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.settings = SettingsStore.load_or_create("settings.json")
        self.settings.data["options"]["enable_test_model"] = True

        provider = CaseStore(3, "provider_003_cases.json")
        provider.add_case("Case one.", ["r1"])
        provider.add_case("Case two.", ["r1"])
        self.master = MasterStore("master_cases.json")
        self.master.merge_provider(provider)
        self.answers = AnswersStore.load_or_create("answers.json", "answer_images")

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def entry(self, model_id):
        return next(m for m in model_catalog(self.settings) if m["model_id"] == model_id)

    def test_slug_rules(self):
        self.assertEqual(model_slug("claude", "claude-opus-4-8"), "claude@claude-opus-4-8")
        self.assertEqual(model_slug("claude", "Claude Fable 5.0"), "claude@claude-fable-5.0")
        self.assertNotEqual(
            model_slug("claude", "claude-opus-4-8"), model_slug("claude", "claude-fable-5.0")
        )
        self.assertEqual(model_slug("openevidence", ""), "openevidence")

    def test_api_model_name_resolves_from_settings_or_default(self):
        entry = self.entry("claude")
        self.assertTrue(entry["model_name"])  # registry default fills in
        self.assertEqual(entry["variant_id"], model_slug("claude", entry["model_name"]))
        self.settings.api_model("claude")["model"] = "claude-opus-4-8"
        entry = self.entry("claude")
        self.assertEqual(entry["model_name"], "claude-opus-4-8")
        self.assertEqual(entry["variant_id"], "claude@claude-opus-4-8")
        self.assertIn("claude-opus-4-8", entry["scored_as"])

    def test_browser_model_name_comes_from_settings_only(self):
        entry = self.entry("openevidence")
        self.assertEqual(entry["model_name"], "")  # user has not set it yet
        self.settings.browser_model("openevidence")["model"] = "OpenEvidence 2026-07"
        entry = self.entry("openevidence")
        self.assertEqual(entry["variant_id"], "openevidence@openevidence-2026-07")
        self.assertEqual(entry["scored_as"], "OpenEvidence (OpenEvidence 2026-07)")

    def test_record_carries_full_identity(self):
        self.settings.api_model("claude")["model"] = "claude-opus-4-8"
        record = new_record(self.master.cases["003-001"], self.entry("claude"), "prompt")
        self.assertEqual(record["model_id"], "claude@claude-opus-4-8")
        self.assertEqual(record["llm_id"], "claude")
        self.assertEqual(record["model_name"], "claude-opus-4-8")
        self.assertIn("claude-opus-4-8", record["model_display_name"])

    def test_two_model_names_on_one_llm_are_separate_work(self):
        # Answer both cases as opus, then switch the model name: the work
        # list must treat the new name as a brand-new, unanswered model.
        self.settings.api_model("claude")["model"] = "claude-opus-4-8"
        opus = self.entry("claude")
        for case_id in ("003-001", "003-002"):
            record = new_record(self.master.cases[case_id], opus, "p")
            record["status"] = "ok"
            self.answers.upsert(record)
        todo, skipped, failed, changed = build_worklist(
            self.master, self.answers, ["003-001", "003-002"], [opus]
        )
        self.assertEqual(skipped, 2)
        self.assertEqual(todo[opus["variant_id"]], [])

        self.settings.api_model("claude")["model"] = "claude-fable-5.0"
        fable = self.entry("claude")
        todo, skipped, failed, changed = build_worklist(
            self.master, self.answers, ["003-001", "003-002"], [fable]
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(todo[fable["variant_id"]], ["003-001", "003-002"])
        # And both identities coexist in the answers store.
        self.assertEqual(len(self.answers.answers), 2)
        self.assertIsNotNone(self.answers.get("003-001", "claude@claude-opus-4-8"))
        self.assertIsNone(self.answers.get("003-001", "claude@claude-fable-5.0"))

    def test_ranker_keeps_variants_separate_with_display_names(self):
        key = {
            "003-001": {
                "A": {"model_id": "claude@claude-opus-4-8", "llm_id": "claude",
                      "model_name": "claude-opus-4-8",
                      "display_name": "Anthropic Claude (claude-opus-4-8)"},
                "B": {"model_id": "claude@claude-fable-5.0", "llm_id": "claude",
                      "model_name": "claude-fable-5.0",
                      "display_name": "Anthropic Claude (claude-fable-5.0)"},
            }
        }
        scores_file = {
            "path": "scores_x.json", "package_id": "pkg_1", "package_name": "x",
            "scorer": "CR",
            "records": [
                {"case_id": "003-001", "label": "A", "score": 2},
                {"case_id": "003-001", "label": "B", "score": 0},
            ],
        }
        matches, warnings = build_matches([scores_file], {"pkg_1": key})
        self.assertEqual(warnings, [])
        model_ids = {m["model_id"] for m in matches}
        self.assertEqual(
            model_ids, {"claude@claude-opus-4-8", "claude@claude-fable-5.0"}
        )
        names = display_map(matches)
        self.assertEqual(names["claude@claude-opus-4-8"], "Anthropic Claude (claude-opus-4-8)")
        self.assertEqual(names["claude@claude-fable-5.0"], "Anthropic Claude (claude-fable-5.0)")


class ChangeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.settings = SettingsStore.load_or_create("settings.json")
        self.settings.data["options"]["enable_test_model"] = True
        provider = CaseStore(3, "provider_003_cases.json")
        provider.add_case("Original text.", ["r1", "r2"])
        self.master = MasterStore("master_cases.json")
        self.master.merge_provider(provider)
        self.case = self.master.cases["003-001"]
        entry = next(
            m for m in model_catalog(self.settings) if m["model_id"] == "testmodel"
        )
        self.record = new_record(self.case, entry, "p")

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_rubric_only_change_does_not_reask_the_models(self):
        # The models never see the rubric, so editing it must not offer
        # to re-run them - only re-grading is affected.
        self.assertFalse(answer_needs_reask(self.record, self.case))
        self.case["rubric"] = ["r1 fixed", "r2"]
        self.assertFalse(answer_needs_reask(self.record, self.case))

    def test_case_text_change_still_reasks(self):
        self.case["case_text"] = "Reworded vignette."
        self.assertTrue(answer_needs_reask(self.record, self.case))

    def test_legacy_records_fall_back_to_the_combined_hash(self):
        legacy = {k: v for k, v in self.record.items() if k != "case_text_sha256"}
        self.assertFalse(answer_needs_reask(legacy, self.case))
        self.case["rubric"] = ["r1 fixed", "r2"]
        # A legacy record cannot tell text from rubric changes, so it
        # conservatively counts as changed.
        self.assertTrue(answer_needs_reask(legacy, self.case))


class FakeUi:
    def __init__(self):
        self.lines = []
        self.told = []
        self.stop_requested = False

    def log(self, message):
        self.lines.append(message)

    def tell(self, title, message):
        self.told.append((title, message))


def api_entry(llm_id, model_name):
    return {
        "model_id": llm_id,
        "kind": "api",
        "model_name": model_name,
        "variant_id": model_slug(llm_id, model_name),
        "display_name": llm_id,
        "scored_as": "{} ({})".format(llm_id, model_name),
    }


class TrustLoginTests(unittest.TestCase):
    def test_continue_anyway_skips_the_login_check(self):
        # The login check is a heuristic; when the user says they are
        # logged in, their word wins and the run proceeds.
        from run_llms import interactive_login

        class StubbornDriver:
            display_name = "GPT-OSS Playground"
            site_id = "gptoss"
            login_url = home_url = "https://example/"

            def wait_until_ready(self, page):
                pass

            def autofill_login(self, page, username, password):
                pass

            def is_logged_in(self, page):
                return False  # the check never believes the user

        class StubPage:
            def goto(self, *args, **kwargs):
                pass

            def is_closed(self):
                return False

            def screenshot(self, **kwargs):
                raise RuntimeError("no screenshots in tests")

        class StubContext:
            def __init__(self, page):
                self.pages = [page]

        class TrustUi(FakeUi):
            def ask_choice(self, title, message, options):
                assert any(key == "trust" for key, _ in options)
                return "trust"

        entry = {}
        page = StubPage()
        self.assertTrue(
            interactive_login(StubbornDriver(), StubContext(page), page, entry, TrustUi())
        )
        self.assertTrue(entry.get("last_login_ok"))

    def test_normal_browser_route_releases_profile_and_requests_reopen(self):
        # Sites that close automated windows during sign-in: the user
        # picks 'Sign in with a normal browser'; the automation window
        # must release the profile first, then the caller reopens.
        import run_llms
        from run_llms import interactive_login

        class Driver:
            display_name = "Doximity Ask"
            site_id = "doximity"
            login_url = home_url = "https://example/"

            def wait_until_ready(self, page):
                pass

            def autofill_login(self, page, username, password):
                pass

            def is_logged_in(self, page):
                return False

        class Page:
            def goto(self, *args, **kwargs):
                pass

            def is_closed(self):
                return False

        class Context:
            def __init__(self, page):
                self.pages = [page]
                self.closed = False

            def close(self):
                self.closed = True

        class NormalUi(FakeUi):
            def ask_choice(self, title, message, options):
                assert any(key == "normal" for key, _ in options)
                return "normal"

        calls = []
        original = run_llms.plain_browser_login
        run_llms.plain_browser_login = lambda site_id, driver, ui: (
            calls.append(site_id) or True
        )
        try:
            page = Page()
            context = Context(page)
            outcome = interactive_login(Driver(), context, page, {}, NormalUi())
        finally:
            run_llms.plain_browser_login = original
        self.assertEqual(outcome, "reopen")
        self.assertTrue(context.closed)  # profile released first
        self.assertEqual(calls, ["doximity"])

    def test_plain_browser_login_launches_the_users_browser(self):
        from run_llms import plain_browser_login

        class Driver:
            display_name = "Doximity Ask"
            site_id = "doximity"
            login_url = "https://www.doximity.com/ask/overview"

        launched = []
        ui = FakeUi()
        import run_llms
        original = run_llms.find_normal_browser
        run_llms.find_normal_browser = lambda: "/fake/chrome"
        try:
            ok = plain_browser_login(
                "doximity", Driver(), ui, launch=lambda cmd: launched.append(cmd)
            )
        finally:
            run_llms.find_normal_browser = original
        self.assertTrue(ok)
        self.assertEqual(len(launched), 1)
        command = launched[0]
        self.assertEqual(command[0], "/fake/chrome")
        self.assertIn("https://www.doximity.com/ask/overview", command)
        self.assertTrue(any("browser_profiles" in part for part in command))
        self.assertTrue(ui.told)  # the user got the step-by-step message

    def test_window_closed_during_signin_requests_reopen(self):
        # The site killed the whole window mid-sign-in: interactive_login
        # reports 'reopen' so the caller relaunches the browser (the saved
        # profile usually kept the session).
        from run_llms import interactive_login

        class Driver:
            display_name = "Doximity Ask"
            site_id = "doximity"
            login_url = home_url = "https://example/"

            def wait_until_ready(self, page):
                pass

            def autofill_login(self, page, username, password):
                pass

            def is_logged_in(self, page):
                return False

        class DeadPage:
            def goto(self, *args, **kwargs):
                raise RuntimeError("target closed")

            def is_closed(self):
                return True

        class DeadContext:
            @property
            def pages(self):
                raise RuntimeError("browser closed")

        class ContinueUi(FakeUi):
            def ask_choice(self, title, message, options):
                return "check"

        outcome = interactive_login(
            Driver(), DeadContext(), DeadPage(), {}, ContinueUi()
        )
        self.assertEqual(outcome, "reopen")


class ParallelApiPhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.settings = SettingsStore.load_or_create("settings.json")
        provider = CaseStore(3, "provider_003_cases.json")
        provider.add_case("Case one.", ["r1"])
        provider.add_case("Case two.", ["r1"])
        self.master = MasterStore("master_cases.json")
        self.master.merge_provider(provider)
        self.answers = AnswersStore.load_or_create("answers.json", "answer_images")
        self.orig_call = run_llms.call_api_model

    def tearDown(self):
        run_llms.call_api_model = self.orig_call
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def run_phase(self, models, todo, ui=None):
        ui = ui or FakeUi()
        run_api_phase(self.master, self.answers, self.settings, models, todo, ui)
        return ui

    def test_two_providers_really_run_at_the_same_time(self):
        # Both workers must arrive at the barrier together; if the models
        # actually ran one after the other, the first would time out
        # waiting for the second and the test would fail.
        barrier = threading.Barrier(2, timeout=10)
        models = [api_entry("claude", "claude-x"), api_entry("gpt", "gpt-x")]
        todo = {m["variant_id"]: ["003-001", "003-002"] for m in models}
        seen_threads = set()

        def fake_call(model_id, entry, prompt, options, log=print, sleep=None):
            seen_threads.add(threading.current_thread().name)
            barrier.wait()  # both providers must be in flight at once
            return {
                "response_text": "Answer from " + model_id,
                "model_requested": model_id,
                "model_reported": model_id,
                "attempts": 1,
            }

        run_llms.call_api_model = fake_call
        ui = self.run_phase(models, todo)
        self.assertEqual(len(seen_threads), 2)
        for model in models:
            for case_id in ("003-001", "003-002"):
                record = self.answers.get(case_id, model["variant_id"])
                self.assertEqual(record["status"], "ok", (case_id, model["variant_id"]))
        # The shared counter reached the total across both workers.
        self.assertTrue(any("[4/4]" in line for line in ui.lines))

    def test_one_provider_aborting_does_not_stop_the_other(self):
        models = [api_entry("claude", "claude-x"), api_entry("gpt", "gpt-x")]
        todo = {m["variant_id"]: ["003-001", "003-002"] for m in models}

        def fake_call(model_id, entry, prompt, options, log=print, sleep=None):
            if model_id == "claude":
                raise ModelAbort("bad key")
            return {
                "response_text": "ok", "model_requested": model_id,
                "model_reported": model_id, "attempts": 1,
            }

        run_llms.call_api_model = fake_call
        ui = self.run_phase(models, todo)
        self.assertIsNone(self.answers.get("003-001", "claude@claude-x"))
        self.assertEqual(self.answers.get("003-002", "gpt@gpt-x")["status"], "ok")
        self.assertTrue(any("skipped for the rest of this run" in line for line in ui.lines))

    def test_unexpected_crash_in_one_worker_is_reported_after_the_others_finish(self):
        models = [api_entry("claude", "claude-x"), api_entry("gpt", "gpt-x")]
        todo = {m["variant_id"]: ["003-001"] for m in models}

        def fake_call(model_id, entry, prompt, options, log=print, sleep=None):
            if model_id == "claude":
                raise ValueError("boom")
            return {
                "response_text": "ok", "model_requested": model_id,
                "model_reported": model_id, "attempts": 1,
            }

        run_llms.call_api_model = fake_call
        with self.assertRaises(RuntimeError) as ctx:
            self.run_phase(models, todo)
        self.assertIn("boom", str(ctx.exception))
        # The healthy provider still finished its work first.
        self.assertEqual(self.answers.get("003-001", "gpt@gpt-x")["status"], "ok")


if __name__ == "__main__":
    unittest.main()
