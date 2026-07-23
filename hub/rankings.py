"""The PI's ranking dashboard - the live Elo fit over every grade.

Reuses the study's proven rating machinery unchanged (rank_llms.py):
every active grade is one match between an (LLM, model name) variant
and a case; 2 = win, 1 = draw, 0 = loss; all ratings are fitted at once
by logistic regression.

Guards, enforced here because the hub IS the master database:
- only grades made under a case's CURRENT rubric version count; older
  ones are listed as awaiting re-grade (the grading pages already
  re-queued them);
- one active grade per (grader, answer) is guaranteed by the schema, so
  accidental double-counting cannot happen; several graders on the same
  answer are intended evidence, reported as inter-rater agreement.
"""

import csv
import io
import json

from flask import Blueprint, Response, flash, g, redirect, render_template, url_for

from case_editor import now_iso
from rank_llms import (
    ELO_CENTER, RESULT_FOR_SCORE, expected_win, fit_ratings,
    multi_scorer_summary,
)
from .auth import login_required, pi_required
from .db import get_db

bp = Blueprint("rankings", __name__)


def build_hub_matches(db):
    """(matches, excluded) - one match per active, current-version grade."""
    rows = db.execute(
        "SELECT grades.score, grades.rubric_version AS grade_version, "
        "answers.variant_id, answers.model_display_name, answers.case_id, "
        "cases.rubric_version AS current_version, users.name AS grader_name "
        "FROM grades "
        "JOIN grading_assignments ga ON ga.id = grades.assignment_id "
        "JOIN users ON users.id = ga.grader_id "
        "JOIN answers ON answers.id = grades.answer_id "
        "JOIN cases ON cases.id = answers.case_id "
        "WHERE grades.superseded = 0 AND cases.deleted = 0 "
        "AND answers.status IN ('ok', 'ok_manual')"
    ).fetchall()
    matches = []
    excluded = []
    for row in rows:
        if row["grade_version"] != row["current_version"]:
            excluded.append(
                "Case {} x {} (graded by {}) is on rubric version {} but the "
                "case is on version {} - awaiting re-grade.".format(
                    row["case_id"], row["model_display_name"] or row["variant_id"],
                    row["grader_name"], row["grade_version"],
                    row["current_version"],
                )
            )
            continue
        if row["score"] is None:
            excluded.append(
                "Case {} x {} (graded by {}) is awaiting completion after a "
                "rubric edit - the grader still owes the new/changed item or "
                "the risk and approach questions.".format(
                    row["case_id"], row["model_display_name"] or row["variant_id"],
                    row["grader_name"],
                )
            )
            continue
        if row["score"] not in RESULT_FOR_SCORE:
            continue
        matches.append({
            "model_id": row["variant_id"],
            "display": row["model_display_name"] or row["variant_id"],
            "case_id": row["case_id"],
            "score": row["score"],
            "result": RESULT_FOR_SCORE[row["score"]],
            "scorer": row["grader_name"],
        })
    return matches, excluded


def ranking_payload(db):
    matches, excluded = build_hub_matches(db)
    if not matches:
        return None, excluded
    llm_ratings, case_ratings, iterations = fit_ratings(matches)
    names = {}
    stats = {}
    for match in matches:
        names.setdefault(match["model_id"], match["display"])
        entry = stats.setdefault(
            match["model_id"], {"n": 0, "sum": 0, "counts": {0: 0, 1: 0, 2: 0}}
        )
        entry["n"] += 1
        entry["sum"] += match["score"]
        entry["counts"][match["score"]] += 1
    ranked = []
    for rank, (model_id, rating) in enumerate(
        sorted(llm_ratings.items(), key=lambda item: -item[1]), start=1
    ):
        entry = stats[model_id]
        ranked.append({
            "rank": rank,
            "model_id": model_id,
            "display": names[model_id],
            "elo": round(rating),
            "n": entry["n"],
            "average": round(entry["sum"] / entry["n"], 2),
            "counts": "{} / {} / {}".format(
                entry["counts"][2], entry["counts"][1], entry["counts"][0]
            ),
            "win_vs_average": round(100 * expected_win(rating, ELO_CENTER)),
        })
    cases = [
        {"case_id": case_id, "elo": round(rating)}
        for case_id, rating in sorted(case_ratings.items(),
                                      key=lambda item: -item[1])
    ]
    payload = {
        "generated_at": now_iso(),
        "matches": len(matches),
        "iterations": iterations,
        "llms": ranked,
        "cases": cases,
        "agreement": multi_scorer_summary(matches, names),
    }
    return payload, excluded


@bp.route("/rankings")
@pi_required
def rankings_page():
    db = get_db()
    payload, excluded = ranking_payload(db)
    snapshots = db.execute(
        "SELECT ranking_snapshots.id, ranking_snapshots.summary, "
        "ranking_snapshots.created_at, users.name AS author FROM "
        "ranking_snapshots JOIN users ON users.id = ranking_snapshots.created_by "
        "ORDER BY ranking_snapshots.id DESC LIMIT 10"
    ).fetchall()
    return render_template(
        "rankings.html", payload=payload, excluded=excluded, snapshots=snapshots,
    )


@bp.route("/rankings.csv")
@pi_required
def rankings_csv():
    payload, _excluded = ranking_payload(get_db())
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["rank", "model", "elo", "graded_answers", "average_score",
                     "twos", "ones", "zeros", "predicted_win_vs_average_case"])
    if payload:
        for row in payload["llms"]:
            twos, ones, zeros = [part.strip() for part in row["counts"].split("/")]
            writer.writerow([
                row["rank"], row["display"], row["elo"], row["n"],
                row["average"], twos, ones, zeros,
                "{}%".format(row["win_vs_average"]),
            ])
    return Response(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=llm_rankings.csv"},
    )


@bp.route("/rankings/snapshot", methods=("POST",))
@pi_required
def snapshot():
    db = get_db()
    payload, excluded = ranking_payload(db)
    if payload is None:
        flash("Nothing to snapshot yet.")
        return redirect(url_for("rankings.rankings_page"))
    summary = "{} matches, {} models, {} cases".format(
        payload["matches"], len(payload["llms"]), len(payload["cases"])
    )
    db.execute(
        "INSERT INTO ranking_snapshots (created_by, summary, results) "
        "VALUES (?, ?, ?)",
        (g.user["id"], summary,
         json.dumps({"payload": payload, "excluded": excluded})),
    )
    db.commit()
    flash("Snapshot saved ({}).".format(summary))
    return redirect(url_for("rankings.rankings_page"))


@bp.route("/my-results")
@login_required
def my_results():
    """A grader's own cases only: per-model averages, no other graders'
    data and no global ranking (that is the PI's)."""
    db = get_db()
    rows = db.execute(
        "SELECT answers.variant_id, answers.model_display_name, grades.score "
        "FROM grades "
        "JOIN grading_assignments ga ON ga.id = grades.assignment_id "
        "JOIN answers ON answers.id = grades.answer_id "
        "JOIN cases ON cases.id = answers.case_id "
        "WHERE grades.superseded = 0 AND grades.score IS NOT NULL "
        "AND cases.owner_id = ? AND ga.grader_id = ?",
        (g.user["id"], g.user["id"]),
    ).fetchall()
    stats = {}
    for row in rows:
        entry = stats.setdefault(row["variant_id"], {
            "display": row["model_display_name"] or row["variant_id"],
            "n": 0, "sum": 0,
        })
        entry["n"] += 1
        entry["sum"] += row["score"]
    results = sorted(
        (
            {"display": entry["display"], "n": entry["n"],
             "average": round(entry["sum"] / entry["n"], 2)}
            for entry in stats.values()
        ),
        key=lambda entry: -entry["average"],
    )
    return render_template("my_results.html", results=results)
