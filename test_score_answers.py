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
from score_answers import (
    AnswerGrader,
    Package,
    PackageError,
    ScoresStore,
    apply_rubric_updates,
    compute_score,
    find_package_zips,
    find_rubric_updates,
    grade_survives_removal,
    load_rubric_updates,
    scores_path_for,
)


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

    def tearDown(self):
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

    # ---- the grading model the window binds to ----

    def test_grader_missed_item_scores_zero_and_skips_extra_questions(self):
        grader = AnswerGrader(["item 1", "item 2"])
        self.assertFalse(grader.complete())
        grader.set_item(0, True)
        grader.set_item(1, False)
        self.assertTrue(grader.complete())          # score already determined
        self.assertFalse(grader.risk_applies())      # questions never apply
        self.assertFalse(grader.poor_applies())
        self.assertEqual(grader.score(), 0)
        self.assertEqual(grader.normalized(), ([True, False], None, None))

    def test_grader_risk_path(self):
        grader = AnswerGrader(["a", "b"])
        grader.set_item(0, True)
        grader.set_item(1, True)
        self.assertTrue(grader.risk_applies())
        self.assertFalse(grader.complete())          # risk not answered yet
        grader.unnecessary_risk = True
        self.assertTrue(grader.complete())
        self.assertFalse(grader.poor_applies())      # skipped once risk = yes
        self.assertEqual(grader.score(), 0)
        self.assertEqual(grader.normalized(), ([True, True], True, None))

    def test_grader_poor_approach_path(self):
        grader = AnswerGrader(["a"])
        grader.set_item(0, True)
        grader.unnecessary_risk = False
        self.assertTrue(grader.poor_applies())
        self.assertFalse(grader.complete())
        grader.poor_approach = True
        self.assertEqual(grader.score(), 1)
        grader.poor_approach = False
        self.assertEqual(grader.score(), 2)

    def test_grader_prefills_from_previous_record(self):
        previous = {
            "rubric_results": [True, True],
            "unnecessary_risk": False,
            "poor_approach": True,
            "comment": "first pass",
        }
        grader = AnswerGrader(["a", "b"], previous)
        self.assertTrue(grader.complete())
        self.assertEqual(grader.score(), 1)
        self.assertEqual(grader.comment, "first pass")

    def test_grader_changing_item_to_missed_drops_followups(self):
        grader = AnswerGrader(["a", "b"])
        grader.set_item(0, True)
        grader.set_item(1, True)
        grader.unnecessary_risk = False
        grader.poor_approach = False
        self.assertEqual(grader.score(), 2)
        grader.set_item(1, False)                    # scorer changes their mind
        self.assertEqual(grader.score(), 0)
        self.assertEqual(grader.normalized(), ([True, False], None, None))

    def test_grader_explanations(self):
        grader = AnswerGrader(["a"])
        self.assertIn("every rubric item", grader.explanation())
        grader.set_item(0, True)
        self.assertIn("unnecessary-risk", grader.explanation())
        grader.unnecessary_risk = False
        self.assertIn("poor-approach", grader.explanation())
        grader.poor_approach = False
        self.assertIn("Score 2", grader.explanation())

    def test_scores_path_naming(self):
        self.assertEqual(scores_path_for("pilot-scorer.zip"), "scores_pilot-scorer.json")
        self.assertEqual(scores_path_for("PKG.ZIP"), "scores_PKG.json")


class RubricUpdateTests(unittest.TestCase):
    """A PI rubric fix arriving at the scorer as a rubric_update file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        manifest = {
            "package_id": "pkg_x",
            "package_name": "x",
            "cases": [
                {"case_id": "003-001", "case_text": "t", "rubric_version": 1,
                 "rubric": ["i1", "i2", "i3"],
                 "answers": [{"label": L, "response_text": "a", "images": []}
                             for L in "ABC"]},
                {"case_id": "003-002", "case_text": "t2", "rubric_version": 1,
                 "rubric": ["b1"],
                 "answers": [{"label": "A", "response_text": "a", "images": []}]},
            ],
        }
        self.package = Package("x.zip", manifest)
        self.scores = ScoresStore("scores_x.json", manifest)
        # A: covered everything except the (to be removed) item 2.
        self.scores.upsert("003-001", "A", [True, False, True], None, None,
                           rubric_version=1)
        # B: covered everything; scored 2.
        self.scores.upsert("003-001", "B", [True, True, True], False, False,
                           rubric_version=1)
        # C: missed item 1 (kept) - still a 0 whatever happens to item 2.
        self.scores.upsert("003-001", "C", [False, True, True], None, None,
                           rubric_version=1)
        self.scores.upsert("003-002", "A", [True], False, False, rubric_version=1)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def removal_update(self):
        return {"003-001": {
            "case_id": "003-001", "rubric": ["i1", "i3"], "rubric_version": 2,
            "from_version": 1,
            "ops": {"removed": [1], "reworded": [], "added": []},
        }}

    def test_grade_survives_removal_rule(self):
        self.assertFalse(grade_survives_removal(
            {"rubric_results": [True, False, True]}, [1]
        ))  # missed ONLY the removed item: outcome now unknown
        self.assertTrue(grade_survives_removal(
            {"rubric_results": [True, True, True]}, [1]
        ))  # covered it: nothing changes
        self.assertTrue(grade_survives_removal(
            {"rubric_results": [False, False, True]}, [1]
        ))  # also missed a KEPT item: still a 0 either way

    def test_removed_item_requeues_only_the_affected_answer(self):
        messages = apply_rubric_updates(self.package, self.scores, self.removal_update())
        self.assertTrue(any("003-001" in m for m in messages))
        # The package copy of the rubric is updated.
        case = self.package.cases[0]
        self.assertEqual(case["rubric"], ["i1", "i3"])
        self.assertEqual(case["rubric_version"], 2)
        # A is re-queued (kept for the audit trail under superseded).
        self.assertIsNone(self.scores.get("003-001", "A"))
        self.assertEqual(len(self.scores.superseded), 1)
        self.assertTrue(self.scores.superseded[0]["superseded"])
        # B and C are carried over: shorter results, same outcome, new version.
        b = self.scores.get("003-001", "B")
        self.assertEqual(b["rubric_results"], [True, True])
        self.assertEqual(b["score"], 2)
        self.assertEqual(b["rubric_version"], 2)
        c = self.scores.get("003-001", "C")
        self.assertEqual(c["rubric_results"], [False, True])
        self.assertEqual(c["score"], 0)
        # The untouched case is untouched.
        self.assertIsNotNone(self.scores.get("003-002", "A"))

    def test_added_or_reworded_item_requeues_every_answer_on_the_case(self):
        updates = {"003-001": {
            "case_id": "003-001", "rubric": ["i1", "i2", "i3", "i4 new"],
            "rubric_version": 2, "from_version": 1,
            "ops": {"removed": [], "reworded": [], "added": [3]},
        }}
        apply_rubric_updates(self.package, self.scores, updates)
        for label in "ABC":
            self.assertIsNone(self.scores.get("003-001", label))
        self.assertEqual(len(self.scores.superseded), 3)
        self.assertIsNotNone(self.scores.get("003-002", "A"))

    def test_update_is_idempotent(self):
        apply_rubric_updates(self.package, self.scores, self.removal_update())
        before = json.dumps(sorted(self.scores.scores.keys()))
        messages = apply_rubric_updates(self.package, self.scores, self.removal_update())
        self.assertEqual(messages, [])  # second application changes nothing
        self.assertEqual(before, json.dumps(sorted(self.scores.scores.keys())))
        self.assertEqual(len(self.scores.superseded), 1)

    def test_unknown_ops_fall_back_to_full_requeue(self):
        updates = {"003-001": {
            "case_id": "003-001", "rubric": ["i1", "i3"], "rubric_version": 3,
            "from_version": None, "ops": None,
        }}
        apply_rubric_updates(self.package, self.scores, updates)
        for label in "ABC":
            self.assertIsNone(self.scores.get("003-001", label))

    def test_flags_and_superseded_round_trip_through_the_file(self):
        self.scores.add_flag("003-001", 1, "i2", "too strict")
        apply_rubric_updates(self.package, self.scores, self.removal_update())
        reloaded = ScoresStore.load_or_create("scores_x.json", self.package.manifest)
        self.assertEqual(reloaded.rubric_flags[0]["note"], "too strict")
        self.assertEqual(len(reloaded.superseded), 1)
        self.assertEqual(reloaded.get("003-001", "B")["rubric_version"], 2)

    def test_update_files_are_found_and_merged(self):
        with open("rubric_update_20260715.json", "w", encoding="utf-8") as f:
            json.dump({"kind": "rubric_update", "cases": [
                {"case_id": "003-001", "rubric": ["i1"], "rubric_version": 2},
            ]}, f)
        with open("rubric_update_20260716.json", "w", encoding="utf-8") as f:
            json.dump({"kind": "rubric_update", "cases": [
                {"case_id": "003-001", "rubric": ["i1 v3"], "rubric_version": 3},
            ]}, f)
        paths = find_rubric_updates()
        self.assertEqual(len(paths), 2)
        updates, problems = load_rubric_updates(paths)
        self.assertEqual(problems, [])
        self.assertEqual(updates["003-001"]["rubric_version"], 3)  # newest wins


if __name__ == "__main__":
    unittest.main()
