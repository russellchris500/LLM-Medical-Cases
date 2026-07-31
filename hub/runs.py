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
from eval_common import model_slug
from llm_api import API_REGISTRY
from llm_browser import SITE_INFO
from .auth import is_grader, login_required, pi_required
from .db import get_db

bp = Blueprint("runs", __name__)

# The LLM SITES a job can ask for: id -> display name (test model
# included so the whole pipeline can be exercised without keys or
# logins). Each site fields any number of MODELS (the llm_models table);
# jobs pick (site, model) pairs and every pair is scored separately.
def llm_choices():
    choices = [("testmodel", "Built-in test model (no keys needed)")]
    for llm_id, entry in API_REGISTRY.items():
        choices.append((llm_id, entry["display_name"] + " (API)"))
    for site_id, info in SITE_INFO.items():
        choices.append((site_id, info["display_name"] + " (browser)"))
    return choices


def ensure_seed_models(db):
    """Seed the model list once: each API provider's default model plus
    the test model. Browser sites start empty on purpose - the grader
    adds the exact model label they run the site with."""
    seeds = [("testmodel", "test-model-1")]
    for llm_id, entry in API_REGISTRY.items():
        seeds.append((llm_id, entry["default_model"]))
    for llm_id, model_name in seeds:
        db.execute(
            "INSERT OR IGNORE INTO llm_models (llm_id, model_name) "
            "VALUES (?, ?)",
            (llm_id, model_name),
        )
    db.commit()


def grouped_models(db):
    """The two-level run-form list: every site in menu order, each with
    its active models."""
    by_site = {}
    for row in db.execute(
        "SELECT * FROM llm_models WHERE active = 1 ORDER BY model_name"
    ).fetchall():
        by_site.setdefault(row["llm_id"], []).append(row)
    return [
        {"llm_id": llm_id, "label": label, "models": by_site.get(llm_id, [])}
        for llm_id, label in llm_choices()
    ]


def selected_model_rows(db, form):
    """The active llm_models rows a posted form selected (llm_model
    checkboxes carry row ids)."""
    wanted = {value for value in form.getlist("llm_model") if value.isdigit()}
    if not wanted:
        return []
    marks = ",".join("?" for _ in wanted)
    return db.execute(
        "SELECT * FROM llm_models WHERE active = 1 AND id IN ({}) "
        "ORDER BY llm_id, model_name".format(marks),
        sorted(wanted),
    ).fetchall()


def parse_llm_entries(llm_ids_json):
    """A job's LLM list, normalized: new-format objects and legacy plain
    site strings -> [(llm_id, model_name-or-None)]. Legacy entries mean
    'whatever model the executing Runner is configured for'."""
    entries = []
    for entry in json.loads(llm_ids_json):
        if isinstance(entry, dict):
            entries.append((entry.get("llm_id", ""),
                            (entry.get("model_name") or "").strip() or None))
        else:
            entries.append((entry, None))
    return entries


def describe_llm_entries(llm_ids_json):
    names = dict(llm_choices())
    parts = []
    for llm_id, model_name in parse_llm_entries(llm_ids_json):
        base = names.get(llm_id, llm_id)
        parts.append("{} [{}]".format(base, model_name)
                     if model_name else base)
    return ", ".join(parts)


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
    ensure_seed_models(db)
    if request.method == "POST":
        if not is_grader(g.user):
            abort(403)
        case_ids = request.form.getlist("case_id")
        # The two-level form posts llm_model row ids; plain llm_id site
        # names are still accepted (legacy clients) and mean "whatever
        # model the executing Runner is configured for".
        model_rows = selected_model_rows(db, request.form)
        known_sites = dict(llm_choices())
        legacy_sites = [site for site in request.form.getlist("llm_id")
                        if site in known_sites]
        llm_entries = [
            {"llm_id": row["llm_id"], "model_name": row["model_name"]}
            for row in model_rows
        ] + legacy_sites
        send_to_pi = request.form.get("assignee") == "pi"
        owned = {
            row["id"]
            for row in db.execute(
                "SELECT id FROM cases WHERE owner_id = ? AND deleted = 0",
                (g.user["id"],),
            )
        }
        if request.form.get("mode") == "missing":
            # One job per selected MODEL, holding every one of this
            # grader's cases that exact model has not answered yet
            # (discarded answers count as unanswered). Case ticks are
            # ignored.
            if not llm_entries:
                error = "Pick at least one model."
            else:
                assignee = g.user
                if send_to_pi:
                    assignee = pi_user(db)
                    if assignee is None:
                        error = "No PI account exists yet."
            if error is None:
                note = request.form.get("note", "").strip()
                created, covered = [], []
                for entry in llm_entries:
                    if isinstance(entry, dict):
                        variant = model_slug(entry["llm_id"],
                                             entry["model_name"])
                        answered = {
                            row["case_id"]
                            for row in db.execute(
                                "SELECT DISTINCT case_id FROM answers "
                                "WHERE variant_id = ? "
                                "AND status IN ('ok', 'ok_manual')",
                                (variant,),
                            )
                        }
                        label = "{} [{}]".format(
                            known_sites[entry["llm_id"]], entry["model_name"]
                        )
                    else:
                        # Legacy site selection: coverage by site.
                        answered = {
                            row["case_id"]
                            for row in db.execute(
                                "SELECT DISTINCT case_id FROM answers "
                                "WHERE llm_id = ? "
                                "AND status IN ('ok', 'ok_manual')",
                                (entry,),
                            )
                        }
                        label = known_sites[entry]
                    missing = sorted(owned - answered)
                    if missing:
                        db.execute(
                            "INSERT INTO run_jobs (requested_by, assigned_to, "
                            "case_ids, llm_ids, note) VALUES (?, ?, ?, ?, ?)",
                            (g.user["id"], assignee["id"], json.dumps(missing),
                             json.dumps([entry]), note),
                        )
                        created.append("{} ({} case(s))".format(
                            label, len(missing)
                        ))
                    else:
                        covered.append(label)
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
                    message = ("Nothing to run - every selected model already "
                               "has an answer for each of your cases.")
                flash(message)
                return redirect(url_for("runs.run_jobs"))
        elif not case_ids or not set(case_ids) <= owned:
            error = "Pick at least one of your own cases."
        elif not llm_entries:
            error = "Pick at least one model."
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
                     json.dumps(llm_entries),
                     request.form.get("note", "").strip()),
                )
                db.commit()
                flash(
                    "Run job created - {} Answers upload to this site "
                    "automatically and go straight into grading.".format(
                        "the PI's computer runs it next."
                        if send_to_pi else
                        "now open 'Run Hub Jobs' on your computer and run it."
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
        entry["llm_list"] = describe_llm_entries(job["llm_ids"])
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
        llm_groups=grouped_models(db), llm_choices=llm_choices(),
        discarded=discarded, error=error,
    )


@bp.route("/runs/models/add", methods=("POST",))
@login_required
def add_model():
    """Any grader can register a model under a site (e.g. a second
    ChatGPT model) - it becomes selectable in every run-job form."""
    if not is_grader(g.user):
        abort(403)
    db = get_db()
    ensure_seed_models(db)
    llm_id = request.form.get("llm_id", "")
    model_name = (request.form.get("model_name") or "").strip()
    if llm_id not in dict(llm_choices()) or not model_name:
        flash("Pick a site and type the model name to add.")
    else:
        db.execute(
            "INSERT INTO llm_models (llm_id, model_name, added_by) "
            "VALUES (?, ?, ?) ON CONFLICT (llm_id, model_name) "
            "DO UPDATE SET active = 1",
            (llm_id, model_name, g.user["id"]),
        )
        db.commit()
        flash("Added {} under {}.".format(
            model_name, dict(llm_choices())[llm_id]
        ))
    return redirect(url_for("runs.run_jobs"))


@bp.route("/runs/models/<int:model_id>/remove", methods=("POST",))
@pi_required
def remove_model(model_id):
    """The PI can retire a model from the pick lists; nothing is
    deleted - old jobs and answers keep referencing it."""
    db = get_db()
    db.execute("UPDATE llm_models SET active = 0 WHERE id = ?", (model_id,))
    db.commit()
    flash("Model hidden from the run-job lists (existing jobs and "
          "answers are untouched).")
    return redirect(url_for("runs.run_jobs"))


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
