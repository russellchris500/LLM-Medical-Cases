"""Tests for eval_common.py. Run with:  python3 -m unittest test_eval_common.py"""

import json
import os
import tempfile
import unittest

from case_editor import CaseStoreError
from eval_common import (
    AnswersStore,
    CaseSetStore,
    SelectionError,
    SettingsStore,
    case_hash,
    parse_selection,
    sort_case_ids,
)


def fake_master(ids):
    """Minimal cases_by_id mapping for selection tests."""
    return {case_id: {"case_id": case_id} for case_id in ids}


class SelectionTests(unittest.TestCase):
    MASTER = fake_master(
        ["003-001", "003-002", "003-003", "003-005", "003-020", "007-001", "007-002", "1000-001"]
    )

    def test_single_ids(self):
        ids, warnings = parse_selection("003-001, 7-2", self.MASTER)
        self.assertEqual(ids, ["003-001", "007-002"])
        self.assertEqual(warnings, [])

    def test_single_missing_warns(self):
        ids, warnings = parse_selection("003-099", self.MASTER)
        self.assertEqual(ids, [])
        self.assertEqual(len(warnings), 1)

    def test_range_with_gap(self):
        ids, warnings = parse_selection("003-001..003-005", self.MASTER)
        self.assertEqual(ids, ["003-001", "003-002", "003-003", "003-005"])
        self.assertEqual(len(warnings), 1)  # 003-004 missing

    def test_range_short_and_word_forms(self):
        for expr in ("003-001..003", "003-001 to 003-003", "3-1..3-3"):
            ids, _ = parse_selection(expr, self.MASTER)
            self.assertEqual(ids, ["003-001", "003-002", "003-003"], expr)

    def test_range_reversed_bounds(self):
        ids, _ = parse_selection("003-003..003-001", self.MASTER)
        self.assertEqual(ids, ["003-001", "003-002", "003-003"])

    def test_cross_provider_range_rejected(self):
        with self.assertRaises(SelectionError):
            parse_selection("003-001..007-002", self.MASTER)

    def test_provider_token(self):
        ids, _ = parse_selection("provider 7", self.MASTER)
        self.assertEqual(ids, ["007-001", "007-002"])
        ids, _ = parse_selection("P7", self.MASTER)
        self.assertEqual(ids, ["007-001", "007-002"])

    def test_all_token(self):
        ids, _ = parse_selection("all", self.MASTER)
        self.assertEqual(len(ids), len(self.MASTER))

    def test_combined_and_deduplicated(self):
        ids, _ = parse_selection("P7, 003-001, 007-001", self.MASTER)
        self.assertEqual(ids, ["003-001", "007-001", "007-002"])

    def test_four_digit_provider(self):
        ids, _ = parse_selection("1000-001", self.MASTER)
        self.assertEqual(ids, ["1000-001"])

    def test_garbage_raises(self):
        with self.assertRaises(SelectionError):
            parse_selection("everything", self.MASTER)

    def test_sorting_is_numeric(self):
        self.assertEqual(
            sort_case_ids(["1000-001", "003-002", "007-001", "003-001"]),
            ["003-001", "003-002", "007-001", "1000-001"],
        )


class CaseSetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "case_sets.json")
        self.store = CaseSetStore.load_or_create(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_save_reload(self):
        self.store.add("pilot", "003-001..003-003", ["003-003", "003-001", "003-002"])
        reloaded = CaseSetStore.load_or_create(self.path)
        self.assertEqual(
            reloaded.sets["pilot"]["case_ids"], ["003-001", "003-002", "003-003"]
        )
        self.assertEqual(reloaded.sets["pilot"]["expression"], "003-001..003-003")

    def test_duplicate_name_rejected_case_insensitively(self):
        self.store.add("pilot", "x", ["003-001"])
        with self.assertRaises(CaseStoreError):
            self.store.add("PILOT", "y", ["003-002"])

    def test_bad_name_rejected(self):
        with self.assertRaises(CaseStoreError):
            self.store.add("bad name!", "x", ["003-001"])

    def test_rename_and_delete(self):
        self.store.add("pilot", "x", ["003-001"])
        self.store.rename("pilot", "pilot-v2")
        self.assertIsNone(self.store.find("pilot"))
        self.assertIsNotNone(self.store.find("pilot-v2"))
        self.store.delete("pilot-v2")
        self.assertEqual(self.store.sets, {})

    def test_resolve_reports_missing(self):
        self.store.add("pilot", "x", ["003-001", "003-002"])
        present, missing = self.store.resolve("pilot", fake_master(["003-001"]))
        self.assertEqual(present, ["003-001"])
        self.assertEqual(missing, ["003-002"])

    def test_frozen_set_does_not_grow(self):
        self.store.add("pilot", "provider 3", ["003-001"])
        present, missing = self.store.resolve(
            "pilot", fake_master(["003-001", "003-002"])
        )
        self.assertEqual(present, ["003-001"])  # 003-002 added later is NOT included


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "settings.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_then_roundtrip(self):
        store = SettingsStore.load_or_create(self.path)
        self.assertEqual(store.api_model("claude")["api_key"], "")
        self.assertEqual(store.option("max_retries"), 5)
        store.api_model("claude")["api_key"] = "sk-test"
        store.save()
        reloaded = SettingsStore.load_or_create(self.path)
        self.assertEqual(reloaded.api_model("claude")["api_key"], "sk-test")
        self.assertEqual(reloaded.option("max_retries"), 5)

    def test_saved_file_is_private(self):
        store = SettingsStore.load_or_create(self.path)
        store.save()
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_missing_keys_filled_from_defaults(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"api_models": {"claude": {"api_key": "sk-x"}}}, f)
        store = SettingsStore.load_or_create(self.path)
        self.assertEqual(store.api_model("claude")["api_key"], "sk-x")
        self.assertEqual(store.api_model("claude")["model"], "")
        self.assertEqual(store.option("answer_stable_seconds"), 20)


class AnswersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "answers.json")
        self.images = os.path.join(self.tmp.name, "answer_images")
        self.store = AnswersStore.load_or_create(self.path, self.images)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, case_id, model_id, status="ok", **extra):
        base = {
            "case_id": case_id,
            "model_id": model_id,
            "status": status,
            "response_text": "answer",
            "images": [],
        }
        base.update(extra)
        return base

    def test_upsert_replaces_and_persists(self):
        self.store.upsert(self.record("003-001", "claude"))
        self.store.upsert(self.record("003-001", "claude", response_text="v2"))
        reloaded = AnswersStore.load_or_create(self.path, self.images)
        self.assertEqual(len(reloaded.answers), 1)
        self.assertEqual(reloaded.get("003-001", "claude")["response_text"], "v2")

    def test_model_ids_and_answered_cases(self):
        self.store.upsert(self.record("003-001", "claude"))
        self.store.upsert(self.record("003-002", "gpt", status="failed"))
        self.store.upsert(self.record("003-003", "openevidence", status="ok_manual"))
        self.assertEqual(self.store.model_ids(), ["claude", "gpt", "openevidence"])
        self.assertEqual(self.store.answered_case_ids(), ["003-001", "003-003"])
        self.assertEqual(
            self.store.answered_case_ids(ok_only=False),
            ["003-001", "003-002", "003-003"],
        )

    def test_forget_removes_answer_and_its_images(self):
        os.makedirs(self.images)
        image = os.path.join(self.images, "003-001_claude_001.png")
        other = os.path.join(self.images, "003-001_gpt_001.png")
        for p in (image, other):
            with open(p, "wb") as f:
                f.write(b"x")
        self.store.upsert(self.record("003-001", "claude"))
        self.store.upsert(self.record("003-001", "gpt"))
        self.assertTrue(self.store.forget("003-001", "claude"))
        self.assertFalse(self.store.forget("003-001", "claude"))  # already gone
        self.assertFalse(os.path.exists(image))
        self.assertTrue(os.path.exists(other))
        reloaded = AnswersStore.load_or_create(self.path, self.images)
        self.assertIsNone(reloaded.get("003-001", "claude"))
        self.assertIsNotNone(reloaded.get("003-001", "gpt"))

    def test_clear_images_only_touches_the_pair(self):
        os.makedirs(self.images)
        keep = os.path.join(self.images, "003-001_gpt_001.png")
        remove = os.path.join(self.images, "003-001_claude_001.png")
        remove2 = os.path.join(self.images, "003-001_claude_page.png")
        for p in (keep, remove, remove2):
            with open(p, "wb") as f:
                f.write(b"png")
        self.store.clear_images("003-001", "claude")
        self.assertTrue(os.path.exists(keep))
        self.assertFalse(os.path.exists(remove))
        self.assertFalse(os.path.exists(remove2))

    def test_case_hash_changes_with_rubric(self):
        a = {"case_text": "text", "rubric": ["one"]}
        b = {"case_text": "text", "rubric": ["one", "two"]}
        self.assertNotEqual(case_hash(a), case_hash(b))
        self.assertEqual(case_hash(a), case_hash({"case_text": "text", "rubric": ["one"]}))


if __name__ == "__main__":
    unittest.main()
