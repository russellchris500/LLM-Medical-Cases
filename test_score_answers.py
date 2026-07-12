"""Tests for score_answers.py. Run with:  python3 -m unittest test_score_answers.py"""

import json
import os
import tempfile
import unittest
import zipfile

from case_editor import CaseStore
from merge_cases import MasterStore
from eval_common import AnswersStore, case_hash
from build_scoring_package import build_package
import score_answers
from score_answers import (
    Package,
    PackageError,
    ScoresStore,
    StopScoring,
    compute_score,
    find_package_zips,
    grade_one,
    scores_path_for,
)


class ScriptedPrompt:
    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, message):
        self.asked.append(message)
        return self.answers.pop(0)


class ScoreAnswersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)

        provider = CaseStore(3, "provider_003_cases.json")
        provider.add_case("Case one text.", ["item 1", "item 2"])
        provider.add_case("Case two text.", ["only item"])
        self.master = MasterStore("master_cases.json")
        self.master.merge_provider(provider)

        answers = AnswersStore.load_or_create("answers.json", "answer_images")
        os.makedirs("answer_images")
        self.image = os.path.join("answer_images", "003-001_siteb_001.png")
        with open(self.image, "wb") as f:
            f.write(b"\x89PNG fake image bytes")
        for case_id in ("003-001", "003-002"):
            answers.upsert(
                {
                    "case_id": case_id,
                    "model_id": "modela",
                    "status": "ok",
                    "response_text": "First system answer for " + case_id,
                    "images": [],
                    "case_sha256": case_hash(self.master.cases[case_id]),
                }
            )
        answers.upsert(
            {
                "case_id": "003-001",
                "model_id": "siteb",
                "status": "ok_manual",
                "response_text": "Illustrated answer",
                "images": [self.image],
                "case_sha256": case_hash(self.master.cases["003-001"]),
            }
        )
        self.zip_path, self.key_path = build_package(
            "pkg", ["003-001", "003-002"], ["modela", "siteb"],
            self.master, answers, warn=lambda *_: None,
        )
        self.package = Package.load(self.zip_path)
        self._orig_prompt = score_answers.prompt

    def tearDown(self):
        score_answers.prompt = self._orig_prompt
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_package_load_and_find(self):
        self.assertEqual(len(self.package.cases), 2)
        self.assertEqual(sum(1 for _ in self.package.all_answers()), 3)
        case, answer = self.package.find("003-001", "b")  # case-insensitive
        self.assertIsNotNone(answer)
        self.assertEqual(case["case_id"], "003-001")
        case, answer = self.package.find("003-001", "Z")
        self.assertIsNone(answer)

    def test_package_rejects_garbage_zip(self):
        with zipfile.ZipFile("junk.zip", "w") as bundle:
            bundle.writestr("other.txt", "hello")
        with self.assertRaises(PackageError):
            Package.load("junk.zip")

    def test_find_package_zips_filters(self):
        with zipfile.ZipFile("junk.zip", "w") as bundle:
            bundle.writestr("other.txt", "hello")
        found = find_package_zips(os.path.dirname(os.path.abspath(self.zip_path)))
        self.assertEqual(found, [os.path.basename(self.zip_path)])

    def test_extract_images(self):
        case, answer = next(
            (c, a) for c, a in self.package.all_answers() if a["images"]
        )
        paths = self.package.extract_images(answer, "extracted")
        self.assertEqual(len(paths), 1)
        with open(paths[0], "rb") as f:
            self.assertEqual(f.read(), b"\x89PNG fake image bytes")

    def test_compute_score_rule(self):
        self.assertEqual(compute_score([True, False], None, None), 0)  # item missed
        self.assertEqual(compute_score([True, True], True, None), 0)   # risk taken
        self.assertEqual(compute_score([True, True], False, True), 1)  # poor approach
        self.assertEqual(compute_score([True, True], False, False), 2)
        self.assertEqual(compute_score([], None, None), 2)  # vacuous but defined

    def test_scores_store_roundtrip_and_resume(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        store.scorer = "Dr Test"
        store.upsert("003-001", "A", [True, False], None, None, "close but wrong")
        reloaded = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        self.assertEqual(reloaded.scorer, "Dr Test")
        record = reloaded.get("003-001", "A")
        self.assertEqual(record["rubric_results"], [True, False])
        self.assertEqual(record["score"], 0)
        self.assertIsNone(record["unnecessary_risk"])
        self.assertIsNone(record["poor_approach"])
        self.assertEqual(record["comment"], "close but wrong")

    def test_upsert_computes_all_three_scores(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        store.upsert("003-001", "A", [True, True], False, False)
        store.upsert("003-001", "B", [True, True], False, True)
        store.upsert("003-002", "A", [True, True], True, None)
        self.assertEqual(store.get("003-001", "A")["score"], 2)
        self.assertEqual(store.get("003-001", "B")["score"], 1)
        self.assertEqual(store.get("003-002", "A")["score"], 0)

    def test_scores_file_for_wrong_package_rejected(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        store.upsert("003-001", "A", [True, True], False, False)
        with self.assertRaises(PackageError):
            ScoresStore.load_or_create(
                "scores_pkg.json", {"package_id": "pkg_other", "package_name": "x"}
            )

    def pick_plain_answer(self):
        return next(
            (c, a) for c, a in self.package.all_answers()
            if c["case_id"] == "003-001" and not a["images"]
        )

    def test_grade_one_missed_item_scores_zero_without_extra_questions(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        case, answer = self.pick_plain_answer()
        scripted = ScriptedPrompt(["y", "n", "needs work"])
        score_answers.prompt = scripted
        grade_one(self.package, store, case, answer)
        record = store.get("003-001", answer["label"])
        self.assertEqual(record["rubric_results"], [True, False])
        self.assertEqual(record["score"], 0)
        self.assertIsNone(record["unnecessary_risk"])  # never asked
        self.assertIsNone(record["poor_approach"])
        self.assertEqual(record["comment"], "needs work")
        # 2 rubric prompts + 1 comment prompt, nothing else.
        self.assertEqual(len(scripted.asked), 3)

    def test_grade_one_risk_scores_zero(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        case, answer = self.pick_plain_answer()
        score_answers.prompt = ScriptedPrompt(["y", "y", "y", "risky plan"])
        grade_one(self.package, store, case, answer)
        record = store.get("003-001", answer["label"])
        self.assertEqual(record["score"], 0)
        self.assertTrue(record["unnecessary_risk"])
        self.assertIsNone(record["poor_approach"])  # skipped once risk = yes

    def test_grade_one_poor_approach_scores_one(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        case, answer = self.pick_plain_answer()
        score_answers.prompt = ScriptedPrompt(["y", "y", "n", "y", ""])
        grade_one(self.package, store, case, answer)
        record = store.get("003-001", answer["label"])
        self.assertEqual(record["score"], 1)
        self.assertFalse(record["unnecessary_risk"])
        self.assertTrue(record["poor_approach"])

    def test_grade_one_clean_answer_scores_two(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        case, answer = self.pick_plain_answer()
        score_answers.prompt = ScriptedPrompt(["y", "y", "n", "n", ""])
        grade_one(self.package, store, case, answer)
        self.assertEqual(store.get("003-001", answer["label"])["score"], 2)

    def test_grade_one_stop_saves_nothing(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        case, answer = self.pick_plain_answer()
        score_answers.prompt = ScriptedPrompt(["y", "s"])
        with self.assertRaises(StopScoring):
            grade_one(self.package, store, case, answer)
        self.assertIsNone(store.get("003-001", answer["label"]))

    def test_regrade_defaults_to_previous_answers(self):
        store = ScoresStore.load_or_create("scores_pkg.json", self.package.manifest)
        case, answer = self.pick_plain_answer()
        score_answers.prompt = ScriptedPrompt(["y", "y", "n", "y", "first pass"])
        grade_one(self.package, store, case, answer)
        # Re-grading with Enter everywhere keeps every judgment and comment.
        score_answers.prompt = ScriptedPrompt(["", "", "", "", ""])
        grade_one(self.package, store, case, answer)
        record = store.get("003-001", answer["label"])
        self.assertEqual(record["rubric_results"], [True, True])
        self.assertFalse(record["unnecessary_risk"])
        self.assertTrue(record["poor_approach"])
        self.assertEqual(record["score"], 1)
        self.assertEqual(record["comment"], "first pass")

    def test_scores_path_naming(self):
        self.assertEqual(scores_path_for("pilot-scorer.zip"), "scores_pilot-scorer.json")
        self.assertEqual(scores_path_for("PKG.ZIP"), "scores_PKG.json")


if __name__ == "__main__":
    unittest.main()
