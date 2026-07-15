"""Tests for rank_llms.py. Run with:  python3 -m unittest test_rank_llms.py"""

import csv
import os
import tempfile
import unittest

from rank_llms import (
    ELO_CENTER,
    build_matches,
    expected_win,
    export_csv,
    fit_ratings,
)
import rank_llms


def match(model_id, case_id, score):
    return {
        "model_id": model_id,
        "case_id": case_id,
        "score": score,
        "result": {0: 0.0, 1: 0.5, 2: 1.0}[score],
        "scorer": "T",
        "package_id": "pkg_x",
    }


def grid(scores_by_model, cases):
    """scores_by_model: model -> score given on every case in cases."""
    return [
        match(model, case, score)
        for model, score in scores_by_model.items()
        for case in cases
    ]


class FitRatingsTests(unittest.TestCase):
    CASES = ["003-{:03d}".format(i) for i in range(1, 11)]

    def test_stronger_model_rates_higher(self):
        matches = grid({"strong": 2, "middling": 1, "weak": 0}, self.CASES)
        llms, cases, _ = fit_ratings(matches)
        self.assertGreater(llms["strong"], llms["middling"])
        self.assertGreater(llms["middling"], llms["weak"])

    def test_all_draws_lands_everyone_at_center(self):
        matches = grid({"a": 1, "b": 1}, self.CASES)
        llms, cases, _ = fit_ratings(matches)
        for rating in list(llms.values()) + list(cases.values()):
            self.assertAlmostEqual(rating, ELO_CENTER, delta=0.5)

    def test_average_case_is_anchored_at_1500(self):
        matches = grid({"strong": 2, "weak": 0}, self.CASES)
        matches += [match("strong", "007-001", 0), match("weak", "007-001", 0)]
        _, cases, _ = fit_ratings(matches)
        mean = sum(cases.values()) / len(cases)
        self.assertAlmostEqual(mean, ELO_CENTER, delta=0.01)

    def test_harder_case_rates_higher(self):
        # Every model aces case E but flunks case H.
        matches = []
        for model in ("m1", "m2", "m3"):
            matches.append(match(model, "003-001", 2))  # easy
            matches.append(match(model, "003-002", 0))  # hard
            matches.append(match(model, "003-003", 1))  # middling
        _, cases, _ = fit_ratings(matches)
        self.assertGreater(cases["003-002"], cases["003-003"])
        self.assertGreater(cases["003-003"], cases["003-001"])

    def test_extreme_records_stay_finite_and_converge(self):
        matches = grid({"perfect": 2, "hopeless": 0}, self.CASES)
        llms, cases, iterations = fit_ratings(matches)
        self.assertLess(iterations, 500)  # actually converged
        for rating in llms.values():
            self.assertTrue(abs(rating - ELO_CENTER) < 2000)

    def test_mixed_scores_converge_quickly(self):
        # Regression test: mixed 0/1/2 records once oscillated against the
        # in-loop re-anchoring and never converged.
        patterns = {
            "a": [2, 2, 2, 2, 2, 2, 1, 1, 2, 0],
            "b": [2, 2, 1, 1, 1, 2, 1, 0, 0, 1],
            "c": [2, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        }
        matches = [
            match(model, case, score)
            for model, row in patterns.items()
            for case, score in zip(self.CASES, row)
        ]
        llms, cases, iterations = fit_ratings(matches)
        self.assertLess(iterations, 100)
        self.assertGreater(llms["a"], llms["b"])
        self.assertGreater(llms["b"], llms["c"])

    def test_fit_is_order_independent(self):
        matches = grid({"a": 2, "b": 1, "c": 0}, self.CASES)
        llms1, cases1, _ = fit_ratings(matches)
        llms2, cases2, _ = fit_ratings(list(reversed(matches)))
        for model in llms1:
            self.assertAlmostEqual(llms1[model], llms2[model], delta=0.05)

    def test_fitted_ratings_predict_the_data(self):
        # 'strong' wins 2 on most cases, draws a couple: its predicted win
        # chance against the average case should be well above 50%.
        scores = {c: 2 for c in self.CASES}
        scores[self.CASES[0]] = 1
        scores[self.CASES[1]] = 1
        matches = [match("strong", c, s) for c, s in scores.items()]
        matches += [match("baseline", c, 1) for c in self.CASES]
        llms, _, _ = fit_ratings(matches)
        self.assertGreater(expected_win(llms["strong"], ELO_CENTER), 0.7)
        self.assertAlmostEqual(expected_win(llms["baseline"], ELO_CENTER), 0.5, delta=0.05)

    def test_more_evidence_moves_rating_further(self):
        few = [match("m", "003-001", 2), match("other", "003-001", 1)]
        many = grid({"m": 2, "other": 1}, self.CASES)
        llms_few, _, _ = fit_ratings(few)
        llms_many, _, _ = fit_ratings(many)
        self.assertGreater(llms_many["m"] - ELO_CENTER, llms_few["m"] - ELO_CENTER)


class BuildMatchesTests(unittest.TestCase):
    def scores_file(self, package_id="pkg_1", records=None):
        return {
            "path": "scores_test.json",
            "package_id": package_id,
            "package_name": "test",
            "scorer": "CR",
            "records": records if records is not None else [
                {"case_id": "003-001", "label": "A", "score": 2},
                {"case_id": "003-001", "label": "B", "score": 0},
                {"case_id": "003-002", "label": "A", "score": 1},
            ],
        }

    KEYS = {
        "pkg_1": {
            "003-001": {"A": {"model_id": "claude"}, "B": {"model_id": "gpt"}},
            "003-002": {"A": {"model_id": "gpt"}},
        }
    }

    def test_join_maps_scores_to_results(self):
        matches, warnings = build_matches([self.scores_file()], self.KEYS)
        self.assertEqual(warnings, [])
        by_pair = {(m["model_id"], m["case_id"]): m for m in matches}
        self.assertEqual(by_pair[("claude", "003-001")]["result"], 1.0)
        self.assertEqual(by_pair[("gpt", "003-001")]["result"], 0.0)
        self.assertEqual(by_pair[("gpt", "003-002")]["result"], 0.5)

    def test_missing_key_file_warns_and_skips(self):
        matches, warnings = build_matches(
            [self.scores_file(package_id="pkg_unknown")], self.KEYS
        )
        self.assertEqual(matches, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("no matching key", warnings[0])

    def test_stale_rubric_version_grades_are_refused(self):
        # Answer A was graded before the PI's rubric fix, B after it (the
        # scorer re-graded B under version 2). A must not be ranked until
        # re-graded - even though the key file still says version 1,
        # because keys go stale the moment a rubric update goes out.
        records = [
            {"case_id": "003-001", "label": "A", "score": 2, "rubric_version": 1},
            {"case_id": "003-001", "label": "B", "score": 0, "rubric_version": 2},
        ]
        matches, warnings = build_matches(
            [self.scores_file(records=records)], self.KEYS,
            rubric_versions={"pkg_1": {"003-001": 1}},
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["model_id"], "gpt")  # B survived
        self.assertEqual(len(warnings), 1)
        self.assertIn("rubric version 1", warnings[0])
        self.assertIn("re-graded", warnings[0])

    def test_master_database_version_is_authoritative(self):
        # Every grade is version 1 and internally consistent, but the
        # master says the rubric is on version 2: nothing ranks until the
        # scorer re-grades.
        matches, warnings = build_matches(
            [self.scores_file()], self.KEYS,
            rubric_versions={"pkg_1": {"003-001": 1, "003-002": 1}},
            current_versions={"003-001": 2},
        )
        case_ids = {m["case_id"] for m in matches}
        self.assertEqual(case_ids, {"003-002"})
        self.assertEqual(len(warnings), 2)  # both 003-001 grades refused

    def test_missing_label_and_bad_score_warn_individually(self):
        records = [
            {"case_id": "003-001", "label": "Z", "score": 2},
            {"case_id": "003-001", "label": "A", "score": None},
            {"case_id": "003-001", "label": "B", "score": 2},
        ]
        matches, warnings = build_matches([self.scores_file(records=records)], self.KEYS)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["model_id"], "gpt")
        self.assertEqual(len(warnings), 2)

    def test_two_scorers_both_count(self):
        files = [self.scores_file(), dict(self.scores_file(), scorer="XY")]
        matches, _ = build_matches(files, self.KEYS)
        self.assertEqual(len(matches), 6)


class ExportTests(unittest.TestCase):
    def test_csv_files_written_with_expected_rows(self):
        matches = [match("claude", "003-001", 2), match("gpt", "003-001", 0)]
        llms, cases, _ = fit_ratings(matches)
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            try:
                paths = export_csv(matches, llms, cases)
                self.assertEqual(len(paths), 3)
                with open(paths[0], newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(rows[0]["model_id"], "claude")  # ranked first
                self.assertEqual(rows[0]["rank"], "1")
                self.assertEqual(rows[1]["model_id"], "gpt")
                with open(paths[2], newline="", encoding="utf-8") as f:
                    match_rows = list(csv.DictReader(f))
                self.assertEqual(len(match_rows), 2)
                self.assertEqual(match_rows[0]["provider_number"], "3")
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
