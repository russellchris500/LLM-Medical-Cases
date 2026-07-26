"""Run jobs: a grader picks cases and LLMs, then either runs them with
their OWN local Runner or sends the job to the PI's.

The job stores which LLMs to use by their llm_id (claude, gpt,
openevidence, ...); whoever executes the job resolves the exact model
name from their own Runner settings, and every uploaded answer records
the full (LLM, model name) identity it was actually run with.
"""

import json

from flask import (
    Blueprint, abort, flash, g, redirect, render_template, request, url_for
)

from case_editor import now_iso
from llm_api import API_REGISTRY
from llm_browser import SITE_INFO
from .auth import is_grader, login_required
from .db import get_db

bp = Blueprint("runs", __name__)

# The LLMs a job can ask for: id -> display name (test model included so
# the whole pipeline can be exercised without keys or logins).
def llm_choices():
    choices = [("testmodel", "Built-in test model (no keys needed)")]
    for llm_id, entry in API_REGISTRY.items():
        choices.append((llm_id, entry["display_name"] + " (API)"))
    for site_id, info in SITE_INFO.items():
        choices.append((site_id, info["display_name"] + " (browser)"))
    return choices


def pi_user(db):
    return db.execute(
        "SELECT * FROM users WHERE role = 'pi' AND disabled = 0 ORDER BY id"
    ).fetchone()


def job_progress(db, job):
    """(answers received, total expected) - expected is cases x LLMs asked
    for; the real variant count can differ (that is fine, this is a gauge)."""
    received = db.execute(
        "SELECT COUNT(*) AS n FROM answers WHERE run_job_id = ?", (job["id"],)
    ).fetchone()["n"]
    expected = len(json.loads(job["case_ids"])) * len(json.loads(job["llm_ids"]))
    return received, expected


@bp.route("/runs", methods=("GET", "POST"))
@login_required
def run_jobs():
    db = get_db()
    error = None
    if request.method == "POST":
        if not is_grader(g.user):
            abort(403)
        case_ids = request.form.getlist("case_id")
        llm_ids = request.form.getlist("llm_id")
        send_to_pi = request.form.get("assignee") == "pi"
        owned = {
            row["id"]
            for row in db.execute(
                "SELECT id FROM cases WHERE owner_id = ? AND deleted = 0",
                (g.user["id"],),
            )
        }
        if request.form.get("mode") == "missing":
            # One job per selected LLM, holding every one of this
            # grader's cases that LLM has not answered yet (discarded
            # answers count as unanswered). Case ticks are ignored.
            if not llm_ids:
                error = "Pick at least one LLM."
            else:
                assignee = g.user
                if send_to_pi:
                    assignee = pi_user(db)
                    if assignee is None:
                        error = "No PI account exists yet."
            if error is None:
                display = dict(llm_choices())
                note = request.form.get("note", "").strip()
                created, covered = [], []
                for llm_id in llm_ids:
                    answered = {
                        row["case_id"]
                        for row in db.execute(
                            "SELECT DISTINCT case_id FROM answers "
                            "WHERE llm_id = ? AND status IN ('ok', 'ok_manual')",
                            (llm_id,),
                        )
                    }
                    missing = sorted(owned - answered)
                    if missing:
                        db.execute(
                            "INSERT INTO run_jobs (requested_by, assigned_to, "
                            "case_ids, llm_ids, note) VALUES (?, ?, ?, ?, ?)",
                            (g.user["id"], assignee["id"], json.dumps(missing),
                             json.dumps([llm_id]), note),
                        )
                        created.append("{} ({} case(s))".format(
                            display.get(llm_id, llm_id), len(missing)
                        ))
                    else:
                        covered.append(display.get(llm_id, llm_id))
                db.commit()
                if created:
                    message = ("Created {} run job(s) covering the unanswered "
                               "cases: {}. {} local Runner will run "
                               "{}.".format(
                                   len(created), "; ".join(created),
                                   "The PI's" if send_to_pi else "Your",
                                   "them" if len(created) > 1 else "it"))
                    if covered:
                        message += (" Already fully answered: {}."
                                    .format(", ".join(covered)))
                else:
                    message = ("Nothing to run - every selected LLM already "
                               "has an answer for each of your cases.")
                flash(message)
                return redirect(url_for("runs.run_jobs"))
        elif not case_ids or not set(case_ids) <= owned:
            error = "Pick at least one of your own cases."
        elif not llm_ids:
            error = "Pick at least one LLM."
        else:
            assignee = g.user
            if send_to_pi:
                assignee = pi_user(db)
                if assignee is None:
                    error = "No PI account exists yet."
            if error is None:
                db.execute(
                    "INSERT INTO run_jobs (requested_by, assigned_to, case_ids, "
                    "llm_ids, note) VALUES (?, ?, ?, ?, ?)",
                    (g.user["id"], assignee["id"], json.dumps(sorted(case_ids)),
                     json.dumps(llm_ids), request.form.get("note", "").strip()),
                )
                db.commit()
                flash(
                    "Run job created - {} will run it with their local Runner "
                    "program.".format(
                        "the PI" if send_to_pi else "you"
                    )
                )
                return redirect(url_for("runs.run_jobs"))

    if g.user["role"] == "pi":
        jobs = db.execute(
            "SELECT run_jobs.*, requester.name AS requester_name, "
            "assignee.name AS assignee_name FROM run_jobs "
            "JOIN users requester ON requester.id = run_jobs.requested_by "
            "JOIN users assignee ON assignee.id = run_jobs.assigned_to "
            "ORDER BY run_jobs.id DESC"
        ).fetchall()
    else:
        jobs = db.execute(
            "SELECT run_jobs.*, requester.name AS requester_name, "
            "assignee.name AS assignee_name FROM run_jobs "
            "JOIN users requester ON requester.id = run_jobs.requested_by "
            "JOIN users assignee ON assignee.id = run_jobs.assigned_to "
            "WHERE requested_by = ? ORDER BY run_jobs.id DESC",
            (g.user["id"],),
        ).fetchall()
    job_rows = []
    for job in jobs:
        entry = dict(job)
        entry["received"], entry["expected"] = job_progress(db, job)
        entry["case_list"] = ", ".join(json.loads(job["case_ids"]))
        entry["llm_list"] = ", ".join(json.loads(job["llm_ids"]))
        job_rows.append(entry)

    # Answers discarded during grading (incomplete captures) that need a
    # fresh run: the case owner sees their own; the PI sees all.
    if g.user["role"] == "pi":
        discarded = db.execute(
            "SELECT answers.*, users.name AS discarder FROM answers "
            "LEFT JOIN users ON users.id = answers.discarded_by "
            "WHERE answers.status = 'discarded' ORDER BY answers.case_id"
        ).fetchall()
    else:
        discarded = db.execute(
            "SELECT answers.*, users.name AS discarder FROM answers "
            "LEFT JOIN users ON users.id = answers.discarded_by "
            "JOIN cases ON cases.id = answers.case_id "
            "WHERE answers.status = 'discarded' AND cases.owner_id = ? "
            "ORDER BY answers.case_id",
            (g.user["id"],),
        ).fetchall()

    my_cases = []
    if is_grader(g.user):
        my_cases = db.execute(
            "SELECT id, case_text FROM cases WHERE owner_id = ? AND deleted = 0 "
            "ORDER BY case_number",
            (g.user["id"],),
        ).fetchall()
    return render_template(
        "runs.html", jobs=job_rows, my_cases=my_cases,
        llm_choices=llm_choices(), discarded=discarded, error=error,
    )


@bp.route("/runs/<int:job_id>/cancel", methods=("POST",))
@login_required
def cancel_job(job_id):
    db = get_db()
    job = db.execute("SELECT * FROM run_jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None or (
        g.user["role"] != "pi" and job["requested_by"] != g.user["id"]
    ):
        abort(404)
    if job["status"] == "open":
        db.execute(
            "UPDATE run_jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (now_iso(), job_id),
        )
        db.commit()
        flash("Cancelled run job #{}.".format(job_id))
    return redirect(url_for("runs.run_jobs"))
