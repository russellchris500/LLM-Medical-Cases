"""LLM-as-a-judge, verified against manual grades.

A judge run pairs one judge model (API sites only) with the answers that
humans have already graded on the chosen cases. Run AI Answers executes
it on the requester's computer (keys never reach the server) and
uploads one judge grade per answer. Here the hub compares each judge
grade with every complete human grade on that answer:

- ACCEPTED means the judge matched the human on EVERY rubric item;
  risk, approach and the final score are compared and reported too;
- every judge grade records exactly which model judged, and whether it
  judged its own model's answer (self) or its own site's other model;
  a run leaves self answers out unless it explicitly allows them;
- editing the rubric leaves earlier judge grades STALE (their rubric
  version moved on) - "Retest judge" creates a fresh run for the case.

Judge grades never enter the main rankings.
"""

import csv
import io
import json

from flask import (
    Blueprint, Response, abort, flash, g, jsonify, redirect,
    render_template, request, url_for
)

from case_editor import now_iso
from eval_common import model_slug
from llm_api import API_REGISTRY
from llm_judge import compare_grades, judge_score
from .auth import is_grader, login_required
from .db import get_db
from .runner_api import token_required
from .runs import llm_choices, pi_user

bp = Blueprint("judge", __name__)

JUDGE_SITES = ["testmodel"] + list(API_REGISTRY)


def judge_choices(db):
    """(llm_models row id, label) for every active model on an API site -
    browser sites cannot act as judges."""
    names = dict(llm_choices())
    rows = db.execute(
        "SELECT * FROM llm_models WHERE active = 1 ORDER BY llm_id, model_name"
    ).fetchall()
    return [
        (row["id"], "{} - {}".format(names.get(row["llm_id"], row["llm_id"]),
                                    row["model_name"]), row)
        for row in rows if row["llm_id"] in JUDGE_SITES
    ]


def visible_case_ids(db):
    if g.user["role"] == "pi":
        rows = db.execute("SELECT id FROM cases WHERE deleted = 0 ORDER BY id")
    else:
        rows = db.execute(
            "SELECT id FROM cases WHERE deleted = 0 AND owner_id = ? ORDER BY id",
            (g.user["id"],),
        )
    return [row["id"] for row in rows]


def human_grades(db, case_id):
    """Complete, active human grades at the case's CURRENT rubric version,
    keyed by answer id."""
    rows = db.execute(
        "SELECT grades.*, users.name AS grader_name FROM grades "
        "JOIN grading_assignments ga ON ga.id = grades.assignment_id "
        "JOIN users ON users.id = ga.grader_id "
        "JOIN answers ON answers.id = grades.answer_id "
        "JOIN cases ON cases.id = answers.case_id "
        "WHERE ga.case_id = ? AND grades.superseded = 0 "
        "AND grades.score IS NOT NULL "
        "AND grades.rubric_version = cases.rubric_version "
        "AND answers.status IN ('ok', 'ok_manual') "
        "ORDER BY grades.answer_id, users.name",
        (case_id,),
    ).fetchall()
    by_answer = {}
    for row in rows:
        by_answer.setdefault(row["answer_id"], []).append(row)
    return by_answer


def latest_judge_grades(db, case_id):
    """The newest judge grade per (judge, answer) on a case (any rubric
    version - staleness is decided by the caller)."""
    rows = db.execute(
        "SELECT judge_grades.* FROM judge_grades "
        "JOIN answers ON answers.id = judge_grades.answer_id "
        "WHERE answers.case_id = ? AND judge_grades.status = 'ok' "
        "ORDER BY judge_grades.id",
        (case_id,),
    ).fetchall()
    latest = {}
    for row in rows:
        latest[(row["judge_variant"], row["answer_id"])] = row
    return latest


def judged_category(row):
    if row["self_judged"]:
        return "self"
    if row["same_site"]:
        return "same_site"
    return "other"


def agreement_report(db, case_ids):
    """Per judge: compared / accepted counts, split by self-judging
    category, stale counts, and the rubric items that disagree most."""
    judges = {}
    for case_id in case_ids:
        case = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        rubric = json.loads(case["rubric"])
        humans = human_grades(db, case_id)
        for (variant, answer_id), judge in latest_judge_grades(db, case_id).items():
            entry = judges.setdefault(variant, {
                "variant": variant, "compared": 0, "accepted": 0, "stale": 0,
                "score_match": 0,
                "by_category": {c: {"compared": 0, "accepted": 0}
                                for c in ("other", "same_site", "self")},
                "item_misses": {},
            })
            if judge["rubric_version"] != case["rubric_version"]:
                entry["stale"] += 1
                continue
            human_rows = humans.get(answer_id, [])
            if not human_rows:
                continue  # the human grade went away (superseded/pending)
            judge_results = json.loads(judge["rubric_results"])
            accepted = True
            score_ok = True
            for human in human_rows:
                verdict = compare_grades(judge_results, json.loads(human["rubric_results"]))
                if not verdict["accepted"]:
                    accepted = False
                    for index in verdict["mismatched"]:
                        key = (case_id, index)
                        miss = entry["item_misses"].setdefault(key, {
                            "case_id": case_id, "index": index,
                            "item": rubric[index] if index < len(rubric) else "?",
                            "count": 0,
                        })
                        miss["count"] += 1
                if human["score"] != judge["score"]:
                    score_ok = False
            category = judged_category(judge)
            entry["compared"] += 1
            entry["by_category"][category]["compared"] += 1
            if accepted:
                entry["accepted"] += 1
                entry["by_category"][category]["accepted"] += 1
            if score_ok:
                entry["score_match"] += 1
    report = []
    for entry in judges.values():
        entry["item_misses"] = sorted(
            entry["item_misses"].values(), key=lambda m: -m["count"]
        )[:8]
        entry["percent"] = (round(100 * entry["accepted"] / entry["compared"])
                            if entry["compared"] else None)
        report.append(entry)
    report.sort(key=lambda e: (-(e["percent"] or -1), e["variant"]))
    return report


def create_judge_run(db, judge_row, case_ids, allow_self, assignee_id):
    versions = {}
    for case_id in case_ids:
        row = db.execute(
            "SELECT rubric_version FROM cases WHERE id = ?", (case_id,)
        ).fetchone()
        versions[case_id] = row["rubric_version"]
    cursor = db.execute(
        "INSERT INTO judge_runs (requested_by, assigned_to, judge_llm_id, "
        "judge_model_name, judge_variant, case_ids, allow_self, "
        "rubric_versions) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (g.user["id"], assignee_id, judge_row["llm_id"], judge_row["model_name"],
         model_slug(judge_row["llm_id"], judge_row["model_name"]),
         json.dumps(sorted(case_ids)), 1 if allow_self else 0,
         json.dumps(versions)),
    )
    db.commit()
    return cursor.lastrowid


def run_answers(db, run):
    """The answers a judge run covers: human-graded answers on its cases,
    minus the judge's own model's answers unless allowed."""
    payload = []
    for case_id in json.loads(run["case_ids"]):
        case = db.execute(
            "SELECT * FROM cases WHERE id = ? AND deleted = 0", (case_id,)
        ).fetchone()
        if case is None:
            continue
        humans = human_grades(db, case_id)
        for answer_id in sorted(humans):
            answer = db.execute("SELECT * FROM answers WHERE id = ?",
                                (answer_id,)).fetchone()
            self_judged = answer["variant_id"] == run["judge_variant"]
            same_site = (not self_judged
                         and answer["llm_id"] == run["judge_llm_id"])
            if self_judged and not run["allow_self"]:
                continue
            payload.append({
                "answer_id": answer_id,
                "case_id": case_id,
                "case_text": case["case_text"],
                "rubric": json.loads(case["rubric"]),
                "rubric_version": case["rubric_version"],
                "response_text": answer["response_text"],
                "image_count": len(json.loads(answer["image_paths"])),
                "self_judged": self_judged,
                "same_site": same_site,
            })
    return payload


# ---------- pages ----------


@bp.route("/judge", methods=("GET", "POST"))
@login_required
def judge_page():
    db = get_db()
    error = None
    choices = judge_choices(db)
    case_ids = visible_case_ids(db)
    if request.method == "POST":
        if not is_grader(g.user) and g.user["role"] != "pi":
            abort(403)
        picked = [c for c in choices if str(c[0]) == request.form.get("judge_model")]
        wanted = [c for c in request.form.getlist("case_id") if c in case_ids]
        if not picked:
            error = "Pick the model that should act as judge."
        elif not wanted:
            error = "Pick at least one case that has human grades."
        else:
            assignee = g.user
            if request.form.get("assignee") == "pi":
                assignee = pi_user(db)
                if assignee is None:
                    error = "No PI account exists yet."
        if error is None:
            run_id = create_judge_run(
                db, picked[0][2], wanted,
                request.form.get("allow_self") == "1", assignee["id"],
            )
            flash("Judge run #{} created - open 'Run AI Answers' on {} "
                  "computer and start it. Results appear here as they "
                  "arrive.".format(
                      run_id, "the PI's" if assignee["id"] != g.user["id"]
                      else "your"))
            return redirect(url_for("judge.judge_page"))
    # Cases with at least one complete human grade are the testable ones.
    cases = []
    for case_id in case_ids:
        humans = human_grades(db, case_id)
        if humans:
            row = db.execute("SELECT case_text FROM cases WHERE id = ?",
                             (case_id,)).fetchone()
            cases.append({"id": case_id, "graded": len(humans),
                          "text": row["case_text"]})
    if g.user["role"] == "pi":
        runs = db.execute(
            "SELECT judge_runs.*, users.name AS requester_name FROM judge_runs "
            "JOIN users ON users.id = judge_runs.requested_by "
            "ORDER BY judge_runs.id DESC"
        ).fetchall()
    else:
        runs = db.execute(
            "SELECT judge_runs.*, users.name AS requester_name FROM judge_runs "
            "JOIN users ON users.id = judge_runs.requested_by "
            "WHERE requested_by = ? ORDER BY judge_runs.id DESC",
            (g.user["id"],),
        ).fetchall()
    run_rows = []
    for run in runs:
        entry = dict(run)
        entry["case_list"] = ", ".join(json.loads(run["case_ids"]))
        entry["done"] = db.execute(
            "SELECT COUNT(*) AS n FROM judge_grades WHERE judge_run_id = ?",
            (run["id"],),
        ).fetchone()["n"]
        entry["expected"] = len(run_answers(db, run))
        run_rows.append(entry)
    return render_template(
        "judge.html", choices=choices, cases=cases, runs=run_rows,
        report=agreement_report(db, case_ids), error=error,
    )


@bp.route("/judge/<case_id>")
@login_required
def judge_case(case_id):
    db = get_db()
    if case_id not in visible_case_ids(db):
        abort(404)
    case = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    rubric = json.loads(case["rubric"])
    humans = human_grades(db, case_id)
    latest = latest_judge_grades(db, case_id)
    judges = sorted({variant for variant, _ in latest})
    answers = []
    # Every answer a human OR a judge has graded - an answer whose human
    # grade went pending after a rubric edit still shows its stale judge
    # grade, so nothing silently disappears.
    answer_ids = sorted(set(humans) | {aid for _, aid in latest})
    for answer_id in answer_ids:
        answer = db.execute("SELECT * FROM answers WHERE id = ?",
                            (answer_id,)).fetchone()
        if answer is None or answer["status"] not in ("ok", "ok_manual"):
            continue
        entry = {
            "answer_id": answer_id,
            "model": answer["model_display_name"] or answer["variant_id"],
            "variant": answer["variant_id"],
            "humans": [
                {"grader": h["grader_name"],
                 "results": json.loads(h["rubric_results"]),
                 "risk": h["unnecessary_risk"], "poor": h["poor_approach"],
                 "score": h["score"]}
                for h in humans.get(answer_id, [])
            ],
            "judges": [],
        }
        for variant in judges:
            judge = latest.get((variant, answer_id))
            if judge is None:
                continue
            results = json.loads(judge["rubric_results"])
            rationale = json.loads(judge["rationale"])
            stale = judge["rubric_version"] != case["rubric_version"]
            mismatched = set()
            # None = no complete human grade to compare against (yet).
            accepted = (not stale) if entry["humans"] else None
            if not stale:
                for h in entry["humans"]:
                    verdict = compare_grades(results, h["results"])
                    if not verdict["accepted"]:
                        accepted = False
                        mismatched.update(verdict["mismatched"])
            entry["judges"].append({
                "variant": variant, "results": results,
                "evidence": rationale.get("evidence", []),
                "risk": judge["unnecessary_risk"], "poor": judge["poor_approach"],
                "risk_reason": rationale.get("risk_reason", ""),
                "poor_reason": rationale.get("poor_reason", ""),
                "score": judge["score"], "stale": stale,
                "accepted": accepted, "mismatched": mismatched,
                "category": judged_category(judge),
                "llm_id": judge["judge_variant"].split("@")[0],
            })
        answers.append(entry)
    # Judges that have graded this case, for the Retest buttons.
    previous = db.execute(
        "SELECT DISTINCT judge_llm_id, judge_model_name, allow_self "
        "FROM judge_runs WHERE case_ids LIKE ? ORDER BY id DESC",
        ("%\"{}\"%".format(case_id),),
    ).fetchall()
    return render_template(
        "judge_case.html", case=case, rubric=rubric, answers=answers,
        judges=judges, previous=previous,
    )


@bp.route("/judge/<case_id>/retest", methods=("POST",))
@login_required
def retest(case_id):
    db = get_db()
    if case_id not in visible_case_ids(db):
        abort(404)
    row = db.execute(
        "SELECT * FROM llm_models WHERE llm_id = ? AND model_name = ?",
        (request.form.get("judge_llm_id"), request.form.get("judge_model_name")),
    ).fetchone()
    if row is None:
        flash("That judge model is no longer in the model list.")
        return redirect(url_for("judge.judge_case", case_id=case_id))
    run_id = create_judge_run(
        db, row, [case_id], request.form.get("allow_self") == "1", g.user["id"]
    )
    flash("Judge run #{} created for case {} under rubric version {} - open "
          "'Run AI Answers' on your computer and start it.".format(
              run_id, case_id,
              db.execute("SELECT rubric_version FROM cases WHERE id = ?",
                         (case_id,)).fetchone()["rubric_version"]))
    return redirect(url_for("judge.judge_case", case_id=case_id))


@bp.route("/judge.csv")
@login_required
def judge_csv():
    db = get_db()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["case", "answer_model", "judge", "self_judged", "same_site",
                     "stale", "human_grader", "accepted", "mismatched_items",
                     "human_score", "judge_score", "judge_items", "human_items"])
    for case_id in visible_case_ids(db):
        case = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        humans = human_grades(db, case_id)
        for (variant, answer_id), judge in latest_judge_grades(db, case_id).items():
            answer = db.execute("SELECT * FROM answers WHERE id = ?",
                                (answer_id,)).fetchone()
            stale = judge["rubric_version"] != case["rubric_version"]
            judge_results = json.loads(judge["rubric_results"])
            for human in humans.get(answer_id, []):
                human_results = json.loads(human["rubric_results"])
                verdict = compare_grades(judge_results, human_results)
                writer.writerow([
                    case_id, answer["model_display_name"] or answer["variant_id"],
                    variant, judge["self_judged"], judge["same_site"], int(stale),
                    human["grader_name"], int(verdict["accepted"] and not stale),
                    " ".join(str(i + 1) for i in verdict["mismatched"]),
                    human["score"], judge["score"],
                    "".join("C" if r else "M" for r in judge_results),
                    "".join("C" if r else "M" for r in human_results),
                ])
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=ai_judge_agreement.csv"})


# ---------- runner endpoints (token auth) ----------


@bp.route("/api/runner/judge-runs")
@token_required
def runner_judge_runs():
    db = get_db()
    runs = db.execute(
        "SELECT judge_runs.*, users.name AS requester_name FROM judge_runs "
        "JOIN users ON users.id = judge_runs.requested_by "
        "WHERE assigned_to = ? AND status = 'open' ORDER BY judge_runs.id",
        (g.runner_user_id,),
    ).fetchall()
    payload = []
    for run in runs:
        done = [row["answer_id"] for row in db.execute(
            "SELECT answer_id FROM judge_grades WHERE judge_run_id = ? "
            "AND status = 'ok'", (run["id"],))]
        payload.append({
            "id": run["id"],
            "requested_by": run["requester_name"],
            "judge_llm_id": run["judge_llm_id"],
            "judge_model_name": run["judge_model_name"],
            "judge_variant": run["judge_variant"],
            "allow_self": bool(run["allow_self"]),
            "answers": run_answers(db, run),
            "done_answer_ids": done,
        })
    return jsonify({"judge_runs": payload})


@bp.route("/api/runner/judge-grades", methods=("POST",))
@token_required
def runner_judge_grade():
    db = get_db()
    body = request.get_json(silent=True) or {}
    run = db.execute(
        "SELECT * FROM judge_runs WHERE id = ? AND assigned_to = ?",
        (body.get("judge_run_id"), g.runner_user_id),
    ).fetchone()
    if run is None:
        return jsonify({"error": "unknown judge run for this token"}), 404
    answer = db.execute("SELECT * FROM answers WHERE id = ?",
                        (body.get("answer_id"),)).fetchone()
    if answer is None or answer["case_id"] not in json.loads(run["case_ids"]):
        return jsonify({"error": "that answer is not part of this judge run"}), 400
    status = body.get("status", "ok")
    results = body.get("results") or []
    risk = poor = score = None
    if status == "ok":
        if not isinstance(results, list) or not results:
            return jsonify({"error": "results are required"}), 400
        results = [bool(r) for r in results]
        score, risk, poor = judge_score({
            "results": results,
            "unnecessary_risk": bool(body.get("unnecessary_risk")),
            "poor_approach": bool(body.get("poor_approach")),
        })
    self_judged = answer["variant_id"] == run["judge_variant"]
    same_site = (not self_judged) and answer["llm_id"] == run["judge_llm_id"]
    rationale = {
        "evidence": body.get("evidence") or [],
        "risk_reason": body.get("risk_reason") or "",
        "poor_reason": body.get("poor_reason") or "",
    }
    db.execute(
        "INSERT INTO judge_grades (judge_run_id, answer_id, judge_variant, "
        "rubric_version, rubric_results, unnecessary_risk, poor_approach, score, "
        "rationale, raw_response, self_judged, same_site, thinking_setting, "
        "status, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?) ON CONFLICT (judge_run_id, answer_id) DO UPDATE SET "
        "rubric_version = excluded.rubric_version, "
        "rubric_results = excluded.rubric_results, "
        "unnecessary_risk = excluded.unnecessary_risk, "
        "poor_approach = excluded.poor_approach, score = excluded.score, "
        "rationale = excluded.rationale, raw_response = excluded.raw_response, "
        "thinking_setting = excluded.thinking_setting, status = excluded.status, "
        "error = excluded.error, created_at = excluded.created_at",
        (run["id"], answer["id"], run["judge_variant"],
         int(body.get("rubric_version") or 0), json.dumps(results),
         None if risk is None else int(risk),
         None if poor is None else int(poor), score,
         json.dumps(rationale), body.get("raw_response") or "",
         1 if self_judged else 0, 1 if same_site else 0,
         body.get("thinking_setting") or "", status,
         body.get("error") or None, now_iso()),
    )
    db.commit()
    return jsonify({"stored": True, "score": score})


@bp.route("/api/runner/judge-runs/<int:run_id>/status", methods=("POST",))
@token_required
def runner_judge_status(run_id):
    db = get_db()
    run = db.execute(
        "SELECT * FROM judge_runs WHERE id = ? AND assigned_to = ?",
        (run_id, g.runner_user_id),
    ).fetchone()
    if run is None:
        return jsonify({"error": "unknown judge run for this token"}), 404
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    if status not in ("done", "failed", "open"):
        return jsonify({"error": "status must be done, failed, or open"}), 400
    db.execute(
        "UPDATE judge_runs SET status = ?, status_note = ?, updated_at = ? "
        "WHERE id = ?",
        (status, body.get("note", ""), now_iso(), run_id),
    )
    db.commit()
    return jsonify({"stored": True})
