"""Tests for the (LLM, model name) scoring identity in run_llms.py.
Run with:  python3 -m unittest test_run_llms.py"""

import os
import tempfile
import unittest

from case_editor import CaseStore
from merge_cases import MasterStore
from eval_common import AnswersStore, SettingsStore, model_slug
from run_llms import build_worklist, model_catalog, new_record
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


if __name__ == "__main__":
    unittest.main()
