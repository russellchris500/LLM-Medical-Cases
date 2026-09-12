"""LLM-as-a-judge: the prompt, the parser, and the comparison rules.

Pure functions, no network - the Runner calls the API with the prompt
built here, the hub compares the parsed result with human grades. Kept
apart so the whole judge can be tested without keys.

Blinding: the prompt never names the model that wrote the answer. The
judge returns one Covered/Missed verdict per rubric item with a quoted
piece of evidence, plus the two closing questions; the 0/1/2 score is
computed by us with the study's rule (compute_score), never by the
judge.

Acceptance: a judge grade is ACCEPTED for an answer only when it
matches the human grader on EVERY rubric item. Risk, approach and the
final score are compared too, but reported separately.
"""

import json
import re

from score_answers import compute_score


class JudgeParseError(ValueError):
    """The judge did not return the JSON we asked for."""


JUDGE_INSTRUCTIONS = """You are an experienced attending physician grading an AI-written answer to a clinical case for a research study. Grade STRICTLY against the rubric below - nothing else counts.

For EACH rubric item decide whether the answer covers it (the substance must actually be present; a vague or generic mention does not count). Quote the shortest passage of the answer that shows the item is covered; if it is not covered write "not found".

Then answer two questions about the answer as a whole:
- unnecessary_risk: did the answer recommend anything that would expose this patient to unnecessary risk?
- poor_approach: even if every item is covered, is the overall approach poor (wrong priorities, unsafe sequencing, important reasoning missing)?

For your information only (do not compute a score yourself): the study scores 0 if any item is missed or unnecessary risk was taken, 1 if everything is covered but the approach is poor, 2 if everything is covered and the approach is sound.

Reply with ONLY a JSON object of exactly this shape and nothing else - no prose, no markdown fences:
{"items": [{"index": 1, "covered": true, "evidence": "..."}, ...],
 "unnecessary_risk": false, "risk_reason": "...",
 "poor_approach": false, "poor_reason": "..."}
"items" must contain one entry per rubric item, in order, index starting at 1."""


def build_judge_prompt(case_text, rubric, answer_text, image_count=0):
    """The full prompt for one (case, answer) pair. Never mentions which
    model wrote the answer."""
    lines = [JUDGE_INSTRUCTIONS, "", "=== CLINICAL CASE ===", case_text.strip(),
             "", "=== RUBRIC (one item per line) ==="]
    for index, item in enumerate(rubric, start=1):
        lines.append("{}. {}".format(index, item))
    lines.append("")
    lines.append("=== THE ANSWER TO GRADE ===")
    if image_count:
        lines.append("(The answer also included {} image(s) that are not shown "
                     "here; grade the text only.)".format(image_count))
    lines.append((answer_text or "").strip() or "(empty answer)")
    lines.append("")
    lines.append("=== YOUR JSON VERDICT ===")
    return "\n".join(lines)


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text):
    text = (text or "").strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JudgeParseError("no JSON object in the judge's reply")
    try:
        return json.loads(text[start:end + 1])
    except ValueError as error:
        raise JudgeParseError("the judge's JSON did not parse: {}".format(error))


def _as_bool(value, what):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return False
    raise JudgeParseError("{} must be true or false".format(what))


def parse_judge_response(text, item_count):
    """-> {"results": [bool...], "evidence": [str...], "unnecessary_risk":
    bool, "risk_reason": str, "poor_approach": bool, "poor_reason": str}
    Raises JudgeParseError when the reply is unusable."""
    data = _extract_json(text)
    items = data.get("items")
    if not isinstance(items, list) or len(items) != item_count:
        raise JudgeParseError(
            "expected {} rubric verdicts, got {}".format(
                item_count, len(items) if isinstance(items, list) else "none"
            )
        )
    results, evidence = [None] * item_count, [""] * item_count
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise JudgeParseError("rubric verdict {} is not an object".format(position + 1))
        index = item.get("index", position + 1)
        try:
            index = int(index)
        except (TypeError, ValueError):
            raise JudgeParseError("rubric verdict has a bad index: {!r}".format(index))
        if not 1 <= index <= item_count or results[index - 1] is not None:
            raise JudgeParseError("rubric verdict index {} is out of place".format(index))
        results[index - 1] = _as_bool(item.get("covered"), "covered")
        evidence[index - 1] = str(item.get("evidence") or "")
    return {
        "results": results,
        "evidence": evidence,
        "unnecessary_risk": _as_bool(data.get("unnecessary_risk", False), "unnecessary_risk"),
        "risk_reason": str(data.get("risk_reason") or ""),
        "poor_approach": _as_bool(data.get("poor_approach", False), "poor_approach"),
        "poor_reason": str(data.get("poor_reason") or ""),
    }


def judge_score(parsed):
    """The study's 0/1/2 rule applied to a parsed verdict. The closing
    questions only apply once every item is covered (as for humans)."""
    results = parsed["results"]
    if not all(results):
        return compute_score(results, None, None), None, None
    risk = bool(parsed["unnecessary_risk"])
    poor = None if risk else bool(parsed["poor_approach"])
    return compute_score(results, risk, poor), risk, poor


def compare_grades(judge_results, human_results):
    """-> {"accepted": bool, "mismatched": [item indexes, 0-based]}.
    Accepted only when EVERY rubric item agrees."""
    if len(judge_results) != len(human_results):
        return {"accepted": False,
                "mismatched": list(range(max(len(judge_results), len(human_results))))}
    mismatched = [
        index for index, (judge, human) in enumerate(zip(judge_results, human_results))
        if bool(judge) != bool(human)
    ]
    return {"accepted": not mismatched, "mismatched": mismatched}


def test_model_verdict(item_count):
    """What the built-in test model 'says' as a judge: everything covered,
    no risk, sound approach - lets the whole pipeline run without keys."""
    return json.dumps({
        "items": [{"index": i + 1, "covered": True, "evidence": "test evidence"}
                  for i in range(item_count)],
        "unnecessary_risk": False, "risk_reason": "",
        "poor_approach": False, "poor_reason": "",
    })
