"""Tests for the Case Editor (Program 1). Run with:  python3 -m unittest test_case_editor.py"""

import json
import os
import tempfile
import unittest

from case_editor import FORMAT_VERSION, CaseStore, CaseStoreError, make_case_id


class CaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "provider_007_cases.json")
        self.store = CaseStore(7, self.path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_case_id_format(self):
        self.assertEqual(make_case_id(7, 1), "007-001")
        self.assertEqual(make_case_id(12, 34), "012-034")

    def test_add_case_assigns_sequential_ids(self):
        first = self.store.add_case("Chest pain case.", ["Diagnoses MI"])
        second = self.store.add_case("Headache case.", ["Diagnoses migraine", "No imaging"])
        self.assertEqual(first["case_id"], "007-001")
        self.assertEqual(second["case_id"], "007-002")
        self.assertEqual(second["rubric"], ["Diagnoses migraine", "No imaging"])

    def test_case_numbers_not_reused_after_delete(self):
        self.store.add_case("Case one.", ["a"])
        self.store.add_case("Case two.", ["b"])
        self.store.delete_case(2)
        third = self.store.add_case("Case three.", ["c"])
        self.assertEqual(third["case_id"], "007-003")

    def test_case_numbers_not_reused_across_save_and_load(self):
        self.store.add_case("Case one.", ["a"])
        self.store.add_case("Case two.", ["b"])
        self.store.delete_case(2)
        self.store.save()
        reloaded = CaseStore.load(self.path)
        third = reloaded.add_case("Case three.", ["c"])
        self.assertEqual(third["case_id"], "007-003")

    def test_rejects_empty_case_text_and_rubric(self):
        with self.assertRaises(CaseStoreError):
            self.store.add_case("   ", ["a"])
        with self.assertRaises(CaseStoreError):
            self.store.add_case("Valid text", [])
        with self.assertRaises(CaseStoreError):
            self.store.add_case("Valid text", ["ok", "   "])

    def test_rubric_items_are_stripped(self):
        case = self.store.add_case("Text", ["  item one  ", "item two"])
        self.assertEqual(case["rubric"], ["item one", "item two"])

    def test_update_case(self):
        self.store.add_case("Original text.", ["original item"])
        updated = self.store.update_case(1, case_text="New text.")
        self.assertEqual(updated["case_text"], "New text.")
        self.assertEqual(updated["rubric"], ["original item"])
        updated = self.store.update_case(1, rubric=["new item 1", "new item 2"])
        self.assertEqual(updated["case_text"], "New text.")
        self.assertEqual(updated["rubric"], ["new item 1", "new item 2"])

    def test_update_missing_case_raises(self):
        with self.assertRaises(CaseStoreError):
            self.store.update_case(99, case_text="x")

    def test_delete_missing_case_raises(self):
        with self.assertRaises(CaseStoreError):
            self.store.delete_case(1)

    def test_save_and_load_round_trip(self):
        self.store.add_case("Case text with\ntwo lines.", ["item 1", "item 2"])
        self.store.add_case("Second case.", ["only item"])
        self.store.save()

        loaded = CaseStore.load(self.path)
        self.assertEqual(loaded.provider_number, 7)
        self.assertEqual(len(loaded.cases), 2)
        self.assertEqual(loaded.cases[0]["case_text"], "Case text with\ntwo lines.")
        self.assertEqual(loaded.cases[0]["rubric"], ["item 1", "item 2"])
        self.assertEqual(loaded.cases[1]["case_id"], "007-002")

    def test_saved_file_is_valid_json_with_expected_shape(self):
        self.store.add_case("Text", ["item"])
        self.store.save()
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["format_version"], FORMAT_VERSION)
        self.assertEqual(data["provider_number"], 7)
        self.assertEqual(data["cases"][0]["case_id"], "007-001")
        self.assertIn("created_at", data["cases"][0])
        self.assertIn("updated_at", data["cases"][0])

    def test_load_rejects_wrong_format_version(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"format_version": 99, "provider_number": 7, "cases": []}, f)
        with self.assertRaises(CaseStoreError):
            CaseStore.load(self.path)

    def test_load_rejects_mismatched_case_id(self):
        bad = {
            "format_version": FORMAT_VERSION,
            "provider_number": 7,
            "cases": [
                {
                    "case_id": "003-001",  # wrong provider prefix
                    "case_number": 1,
                    "case_text": "Text",
                    "rubric": ["item"],
                }
            ],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        with self.assertRaises(CaseStoreError):
            CaseStore.load(self.path)

    def test_load_rejects_duplicate_case_numbers(self):
        case = {
            "case_id": "007-001",
            "case_number": 1,
            "case_text": "Text",
            "rubric": ["item"],
        }
        bad = {"format_version": FORMAT_VERSION, "provider_number": 7, "cases": [case, dict(case)]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        with self.assertRaises(CaseStoreError):
            CaseStore.load(self.path)

    def test_load_rejects_invalid_json(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(CaseStoreError):
            CaseStore.load(self.path)


if __name__ == "__main__":
    unittest.main()
