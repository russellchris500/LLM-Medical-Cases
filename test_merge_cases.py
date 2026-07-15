"""Tests for the Case Merger (Program 2). Run with:  python3 -m unittest test_merge_cases.py"""

import json
import os
import tempfile
import unittest

from case_editor import FORMAT_VERSION, CaseStore, CaseStoreError
from merge_cases import (
    MasterStore,
    read_rubric_flags,
    rubric_update_payload,
)


def make_provider(tmpdir, provider_number, cases):
    """Build a saved provider CaseStore with the given (text, rubric) cases."""
    path = os.path.join(tmpdir, "provider_{:03d}_cases.json".format(provider_number))
    store = CaseStore(provider_number, path)
    for text, rubric in cases:
        store.add_case(text, rubric)
    store.save()
    return store


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.master_path = os.path.join(self.tmpdir, "master_cases.json")
        self.master = MasterStore(self.master_path)

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_merge_two_providers(self):
        p3 = make_provider(self.tmpdir, 3, [("Case A", ["a1"]), ("Case B", ["b1", "b2"])])
        p7 = make_provider(self.tmpdir, 7, [("Case C", ["c1"])])

        report3 = self.master.merge_provider(p3)
        report7 = self.master.merge_provider(p7)

        self.assertEqual(report3["added"], ["003-001", "003-002"])
        self.assertEqual(report7["added"], ["007-001"])
        self.assertEqual(set(self.master.cases), {"003-001", "003-002", "007-001"})
        self.assertEqual(self.master.cases["007-001"]["provider_number"], 7)

    def test_reimport_unchanged_file(self):
        p3 = make_provider(self.tmpdir, 3, [("Case A", ["a1"])])
        self.master.merge_provider(p3)
        report = self.master.merge_provider(p3)
        self.assertEqual(report["unchanged"], ["003-001"])
        self.assertEqual(report["added"], [])
        self.assertEqual(report["updated"], [])

    def test_newer_incoming_version_wins_by_default(self):
        p3 = make_provider(self.tmpdir, 3, [("Original text", ["a1"])])
        self.master.merge_provider(p3)
        # Simulate the provider editing the case later.
        p3.update_case(1, case_text="Revised text")
        p3.cases[0]["updated_at"] = "2030-01-01T00:00:00Z"

        report = self.master.merge_provider(p3)
        self.assertEqual(report["updated"], ["003-001"])
        self.assertEqual(self.master.cases["003-001"]["case_text"], "Revised text")

    def test_older_incoming_version_keeps_master(self):
        p3 = make_provider(self.tmpdir, 3, [("Newest text", ["a1"])])
        self.master.merge_provider(p3)
        self.master.cases["003-001"]["updated_at"] = "2030-01-01T00:00:00Z"
        p3.cases[0]["case_text"] = "Stale text"
        p3.cases[0]["updated_at"] = "2020-01-01T00:00:00Z"

        report = self.master.merge_provider(p3)
        self.assertEqual(report["kept_existing"], ["003-001"])
        self.assertEqual(self.master.cases["003-001"]["case_text"], "Newest text")

    def test_conflict_callback_decides(self):
        p3 = make_provider(self.tmpdir, 3, [("Original", ["a1"])])
        self.master.merge_provider(p3)
        p3.cases[0]["case_text"] = "Changed"

        report = self.master.merge_provider(p3, on_conflict=lambda old, new: False)
        self.assertEqual(report["kept_existing"], ["003-001"])
        report = self.master.merge_provider(p3, on_conflict=lambda old, new: True)
        self.assertEqual(report["updated"], ["003-001"])
        self.assertEqual(self.master.cases["003-001"]["case_text"], "Changed")

    def test_missing_case_kept_by_default(self):
        p3 = make_provider(self.tmpdir, 3, [("Case A", ["a1"]), ("Case B", ["b1"])])
        self.master.merge_provider(p3)
        p3.delete_case(2)

        report = self.master.merge_provider(p3)
        self.assertEqual(report["missing_kept"], ["003-002"])
        self.assertIn("003-002", self.master.cases)

    def test_missing_case_removed_when_callback_agrees(self):
        p3 = make_provider(self.tmpdir, 3, [("Case A", ["a1"]), ("Case B", ["b1"])])
        self.master.merge_provider(p3)
        p3.delete_case(2)

        report = self.master.merge_provider(p3, on_missing=lambda case: True)
        self.assertEqual(report["removed"], ["003-002"])
        self.assertNotIn("003-002", self.master.cases)
        self.assertIn("003-001", self.master.cases)

    def test_missing_check_ignores_other_providers(self):
        p3 = make_provider(self.tmpdir, 3, [("Case A", ["a1"])])
        p7 = make_provider(self.tmpdir, 7, [("Case C", ["c1"])])
        self.master.merge_provider(p3)

        report = self.master.merge_provider(p7, on_missing=lambda case: True)
        self.assertEqual(report["removed"], [])
        self.assertIn("003-001", self.master.cases)

    def test_save_and_load_round_trip(self):
        p3 = make_provider(self.tmpdir, 3, [("Case A", ["a1"])])
        p7 = make_provider(self.tmpdir, 7, [("Case C", ["c1", "c2"])])
        self.master.merge_provider(p3)
        self.master.merge_provider(p7)
        self.master.save()

        loaded = MasterStore.load(self.master_path)
        self.assertEqual(set(loaded.cases), {"003-001", "007-001"})
        self.assertEqual(loaded.cases["007-001"]["rubric"], ["c1", "c2"])
        self.assertEqual(loaded.cases["007-001"]["provider_number"], 7)

    def test_saved_master_is_sorted_by_provider_then_case(self):
        p7 = make_provider(self.tmpdir, 7, [("Case C", ["c1"])])
        p3 = make_provider(self.tmpdir, 3, [("Case A", ["a1"]), ("Case B", ["b1"])])
        self.master.merge_provider(p7)
        self.master.merge_provider(p3)
        self.master.save()

        with open(self.master_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(
            [c["case_id"] for c in data["cases"]], ["003-001", "003-002", "007-001"]
        )
        self.assertEqual(data["format_version"], FORMAT_VERSION)

    def test_load_rejects_missing_provider_number(self):
        bad = {
            "format_version": FORMAT_VERSION,
            "cases": [
                {
                    "case_id": "003-001",
                    "case_number": 1,
                    "case_text": "Text",
                    "rubric": ["item"],
                }
            ],
        }
        with open(self.master_path, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        with self.assertRaises(CaseStoreError):
            MasterStore.load(self.master_path)

    def test_load_rejects_wrong_format_version(self):
        with open(self.master_path, "w", encoding="utf-8") as f:
            json.dump({"format_version": 99, "cases": []}, f)
        with self.assertRaises(CaseStoreError):
            MasterStore.load(self.master_path)

    def test_load_or_create_on_missing_file(self):
        store = MasterStore.load_or_create(self.master_path)
        self.assertEqual(store.cases, {})


class RubricEditTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.master = MasterStore(os.path.join(self.tmpdir, "master_cases.json"))
        provider = make_provider(
            self.tmpdir, 3, [("Case A", ["i1", "i2", "i3"]), ("Case B", ["b1"])]
        )
        self.master.merge_provider(provider)

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_edit_bumps_version_and_keeps_history(self):
        case = self.master.edit_rubric(
            "003-001", ["i1", "i3"],
            ops={"removed": [1], "reworded": [], "added": []},
            reason="item 2 was too hard",
        )
        self.assertEqual(case["rubric_version"], 2)
        self.assertEqual(case["rubric"], ["i1", "i3"])
        entry = case["rubric_history"][-1]
        self.assertEqual(entry["from_version"], 1)
        self.assertEqual(entry["old_rubric"], ["i1", "i2", "i3"])
        self.assertEqual(entry["ops"]["removed"], [1])
        self.assertEqual(entry["reason"], "item 2 was too hard")
        # Round-trips through save/load.
        self.master.save()
        reloaded = MasterStore.load(self.master.path)
        self.assertEqual(reloaded.cases["003-001"]["rubric_version"], 2)
        self.assertEqual(len(reloaded.cases["003-001"]["rubric_history"]), 1)

    def test_edit_rejects_empty_and_ignores_no_change(self):
        with self.assertRaises(CaseStoreError):
            self.master.edit_rubric("003-001", [])
        case = self.master.edit_rubric("003-001", ["i1", "i2", "i3"])
        self.assertEqual(case["rubric_version"], 1)  # unchanged rubric, no bump

    def test_provider_re_merge_keeps_versions_monotonic(self):
        # PI fixes the rubric (v2); the provider, unaware, sends their own
        # newer edit. The merged case must not fall back to a lower version.
        self.master.edit_rubric("003-001", ["i1", "i3"], reason="PI fix")
        provider = CaseStore.load(
            os.path.join(self.tmpdir, "provider_003_cases.json")
        )
        provider.update_case(1, rubric=["i1", "i2 clarified", "i3"])  # their v2
        provider.save()
        report = self.master.merge_provider(provider, on_conflict=lambda a, b: True)
        self.assertIn("003-001", report["updated"])
        self.assertEqual(self.master.cases["003-001"]["rubric_version"], 3)

    def test_rubric_update_payload_carries_ops(self):
        self.master.edit_rubric(
            "003-001", ["i1", "i3"],
            ops={"removed": [1], "reworded": [], "added": []}, reason="too hard",
        )
        payload = rubric_update_payload(self.master, ["003-001"])
        self.assertEqual(payload["kind"], "rubric_update")
        entry = payload["cases"][0]
        self.assertEqual(entry["case_id"], "003-001")
        self.assertEqual(entry["rubric"], ["i1", "i3"])
        self.assertEqual(entry["rubric_version"], 2)
        self.assertEqual(entry["from_version"], 1)
        self.assertEqual(entry["ops"]["removed"], [1])

    def test_read_rubric_flags_from_scores_files(self):
        path = os.path.join(self.tmpdir, "scores_pkg.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "format_version": 1, "scorer": "CR",
                "rubric_flags": [
                    {"case_id": "003-001", "item_index": 1,
                     "item_text": "i2", "note": "too strict"},
                ],
            }, f)
        flags, problems = read_rubric_flags([path])
        self.assertEqual(problems, [])
        self.assertEqual(flags[0]["case_id"], "003-001")
        self.assertEqual(flags[0]["scorer"], "CR")
        self.assertEqual(flags[0]["note"], "too strict")


if __name__ == "__main__":
    unittest.main()
