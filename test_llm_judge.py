"""Tests for the LLM-as-a-judge prompt, parser, scoring and comparison.
No network, no Flask.  Run with:  python3 -m unittest test_llm_judge.py"""

import json
import unittest

import llm_judge
from llm_judge import (
    JudgeParseError, build_judge_prompt, compare_grades, judge_score,
    parse_judge_response, test_model_verdict,
)


class PromptTests(unittest.TestCase):
    def test_prompt_is_blinded_and_lists_rubric(self):
        prompt = build_judge_prompt(
            "Chest pain.", ["orders ECG", "gives aspirin"],
            "Get an ECG now and give aspirin.", image_count=2,
        )
        self.assertIn("1. orders ECG", prompt)
        self.assertIn("2. gives aspirin", prompt)
        self.assertIn("Get an ECG now", prompt)
        self.assertIn("2 image(s)", prompt)
        for forbidden in ("claude", "gpt", "openai", "anthropic"):
            self.assertNotIn(forbidden, prompt.lower())
        self.assertIn("ONLY a JSON object", prompt)


class ParseTests(unittest.TestCase):
    def verdict(self, **overrides):
        data = {
            "items": [{"index": 1, "covered": True, "evidence": "ECG now"},
                      {"index": 2, "covered": False, "evidence": "not found"}],
            "unnecessary_risk": False, "risk_reason": "",
            "poor_approach": False, "poor_reason": "",
        }
        data.update(overrides)
        return json.dumps(data)

    def test_bare_and_fenced_json(self):
        for text in (self.verdict(),
                     "Here you go:\n```json\n" + self.verdict() + "\n```",
                     "Sure. " + self.verdict() + " Done."):
            parsed = parse_judge_response(text, 2)
            self.assertEqual(parsed["results"], [True, False])
            self.assertEqual(parsed["evidence"], ["ECG now", "not found"])

    def test_out_of_order_and_string_booleans(self):
        text = json.dumps({
            "items": [{"index": 2, "covered": "yes", "evidence": "x"},
                      {"index": 1, "covered": "no", "evidence": "y"}],
            "unnecessary_risk": "true", "poor_approach": False,
        })
        parsed = parse_judge_response(text, 2)
        self.assertEqual(parsed["results"], [False, True])
        self.assertTrue(parsed["unnecessary_risk"])

    def test_rejects_wrong_count_and_garbage(self):
        with self.assertRaises(JudgeParseError):
            parse_judge_response(self.verdict(), 3)
        with self.assertRaises(JudgeParseError):
            parse_judge_response("I cannot grade this.", 2)
        with self.assertRaises(JudgeParseError):
            parse_judge_response(json.dumps({"items": [
                {"index": 1, "covered": True}, {"index": 1, "covered": True}]}), 2)

    def test_test_model_verdict_round_trips(self):
        parsed = parse_judge_response(test_model_verdict(3), 3)
        self.assertEqual(parsed["results"], [True, True, True])


class ScoreAndCompareTests(unittest.TestCase):
    def test_score_rule(self):
        base = {"unnecessary_risk": False, "poor_approach": False}
        self.assertEqual(judge_score(dict(base, results=[True, False])),
                         (0, None, None))
        self.assertEqual(judge_score(dict(base, results=[True, True],
                                          unnecessary_risk=True)),
                         (0, True, None))
        self.assertEqual(judge_score(dict(base, results=[True, True],
                                          poor_approach=True)),
                         (1, False, True))
        self.assertEqual(judge_score(dict(base, results=[True, True])),
                         (2, False, False))

    def test_accepted_only_when_every_item_matches(self):
        self.assertEqual(compare_grades([True, False], [True, False]),
                         {"accepted": True, "mismatched": []})
        self.assertEqual(compare_grades([True, True], [True, False]),
                         {"accepted": False, "mismatched": [1]})
        self.assertFalse(compare_grades([True], [True, False])["accepted"])


if __name__ == "__main__":
    unittest.main()
