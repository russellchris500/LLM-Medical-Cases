"""The HTTP API the local Runner program talks to.

Authentication: a per-device runner token (created on the hub's
"Runner tokens" page, shown once) sent as  Authorization: Bearer <token>.
Only the token's owner's jobs are visible; API keys and site logins never
reach the hub - the Runner keeps them local and uploads only results.

Endpoints:
    GET  /api/runner/jobs                    open jobs for this user
    POST /api/runner/answers                 one answer (multipart: fields +
                                             optional answer_html + images)
    POST /api/runner/jobs/<id>/status        {"status": "done"|"failed", "note": ...}
"""

import functools
import hashlib
import json
import os
import secrets

from flask import (
    Blueprint, current_app, flash, g, jsonify, redirect, render_template,
    request, url_for
)

from case_editor import now_iso
from build_scoring_package import SELF_ID_STRINGS
from .auth import login_required
from .db import get_db

bp = Blueprint("runner_api", __name__)


def hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_required(view):
    @functools.wraps(view)
    def wrapped(**kwargs):
        header = request.headers.get("Authorization", "")
        token = header[7:].strip() if header.startswith("Bearer ") else ""
        row = None
        if token:
            db = get_db()
            row = db.execute(
                "SELECT runner_tokens.*, users.name AS user_name, users.role "
                "FROM runner_tokens JOIN users ON users.id = runner_tokens.user_id "
                "WHERE token_hash = ? AND revoked = 0 AND users.disabled = 0",
                (hash_token(token),),
            ).fetchone()
        if row is None:
            return jsonify({"error": "a valid runner token is required"}), 401
        db = get_db()
        db.execute(
            "UPDATE runner_tokens SET last_seen = ? WHERE id = ?",
            (now_iso(), row["id"]),
        )
        db.commit()
        g.runner_user_id = row["user_id"]
        return view(**kwargs)
    return wrapped


# ---------- token management page (browser, session auth) ----------


@bp.route("/runner-tokens", methods=("GET", "POST"))
@login_required
def runner_tokens():
    db = get_db()
    new_token = None
    if request.method == "POST":
        if request.form.get("revoke"):
            db.execute(
                "UPDATE runner_tokens SET revoked = 1 "
                "WHERE id = ? AND user_id = ?",
                (request.form["revoke"], g.user["id"]),
            )
            db.commit()
            flash("Token revoked.")
            return redirect(url_for("runner_api.runner_tokens"))
        new_token = secrets.token_urlsafe(32)
        db.execute(
            "INSERT INTO runner_tokens (user_id, token_hash, label) "
            "VALUES (?, ?, ?)",
            (g.user["id"], hash_token(new_token),
             request.form.get("label", "").strip() or "my computer"),
        )
        db.commit()
    rows = db.execute(
        "SELECT * FROM runner_tokens WHERE user_id = ? AND revoked = 0 "
        "ORDER BY id DESC",
        (g.user["id"],),
    ).fetchall()
    return render_template("runner_tokens.html", tokens=rows, new_token=new_token)


# ---------- runner endpoints (token auth) ----------


@bp.route("/api/runner/jobs")
@token_required
def jobs():
    db = get_db()
    rows = db.execute(
        "SELECT run_jobs.*, requester.name AS requester_name FROM run_jobs "
        "JOIN users requester ON requester.id = run_jobs.requested_by "
        "WHERE assigned_to = ? AND status = 'open' ORDER BY run_jobs.id",
        (g.runner_user_id,),
    ).fetchall()
    payload = []
    for job in rows:
        case_ids = json.loads(job["case_ids"])
        marks = ",".join("?" for _ in case_ids)
        cases = db.execute(
            "SELECT id, case_text, rubric_version, owner_id FROM cases "
            "WHERE id IN ({}) AND deleted = 0".format(marks),
            case_ids,
        ).fetchall()
        payload.append({
            "id": job["id"],
            "requested_by": job["requester_name"],
            "note": job["note"],
            "llm_ids": json.loads(job["llm_ids"]),
            "cases": [
                {
                    "case_id": c["id"],
                    "case_text": c["case_text"],
                    "rubric_version": c["rubric_version"],
                    "owner_id": c["owner_id"],
                }
                for c in sorted(cases, key=lambda c: c["id"])
            ],
        })
    return jsonify({"jobs": payload})


def store_upload_file(upload, subdir, name):
    directory = os.path.join(current_app.config["ANSWER_DIR"], subdir)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    upload.save(path)
    return os.path.relpath(path, current_app.config["ANSWER_DIR"])


@bp.route("/api/runner/answers", methods=("POST",))
@token_required
def upload_answer():
    db = get_db()
    form = request.form
    job = db.execute(
        "SELECT * FROM run_jobs WHERE id = ? AND assigned_to = ?",
        (form.get("run_job_id"), g.runner_user_id),
    ).fetchone()
    if job is None:
        return jsonify({"error": "unknown run job for this token"}), 404
    case = db.execute(
        "SELECT * FROM cases WHERE id = ? AND deleted = 0", (form.get("case_id"),)
    ).fetchone()
    if case is None or case["id"] not in json.loads(job["case_ids"]):
        return jsonify({"error": "that case is not part of this job"}), 400
    variant_id = (form.get("variant_id") or "").strip()
    if not variant_id:
        return jsonify({"error": "variant_id is required"}), 400

    subdir = "{}_{}".format(case["id"], g.runner_user_id)
    safe_variant = "".join(
        ch if ch.isalnum() or ch in "-._@" else "-" for ch in variant_id
    )
    image_paths = []
    for index, upload in enumerate(request.files.getlist("images"), start=1):
        extension = os.path.splitext(upload.filename or "")[1][:8] or ".png"
        image_paths.append(store_upload_file(
            upload, subdir, "{}_{:03d}{}".format(safe_variant, index, extension)
        ))
    html_path = None
    if "answer_html" in request.files:
        html_path = store_upload_file(
            request.files["answer_html"], subdir, safe_variant + "_answer.html"
        )

    text = form.get("response_text", "")
    hits = sorted({s for s in SELF_ID_STRINGS if s.lower() in text.lower()})
    self_id_warning = ", ".join(hits)

    db.execute(
        "INSERT INTO answers (case_id, run_job_id, run_by, llm_id, model_name, "
        "variant_id, model_display_name, response_text, answer_html_path, "
        "image_paths, thinking_setting, model_reported, deep_thinking, status, "
        "error, case_text_sha256, rubric_version_at_run, run_by_owner, "
        "self_id_warning, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (case_id, variant_id, run_by) DO UPDATE SET "
        "run_job_id = excluded.run_job_id, llm_id = excluded.llm_id, "
        "model_name = excluded.model_name, "
        "model_display_name = excluded.model_display_name, "
        "response_text = excluded.response_text, "
        "answer_html_path = excluded.answer_html_path, "
        "image_paths = excluded.image_paths, "
        "thinking_setting = excluded.thinking_setting, "
        "model_reported = excluded.model_reported, "
        "deep_thinking = excluded.deep_thinking, status = excluded.status, "
        "error = excluded.error, case_text_sha256 = excluded.case_text_sha256, "
        "rubric_version_at_run = excluded.rubric_version_at_run, "
        "run_by_owner = excluded.run_by_owner, "
        "self_id_warning = excluded.self_id_warning, "
        "created_at = excluded.created_at, "
        # A fresh run replaces (and revives) a discarded answer.
        "discarded_by = NULL, discarded_reason = ''",
        (
            case["id"], job["id"], g.runner_user_id,
            form.get("llm_id", ""), form.get("model_name", ""), variant_id,
            form.get("model_display_name", ""), text, html_path,
            json.dumps(image_paths), form.get("thinking_setting", ""),
            form.get("model_reported", ""),
            1 if form.get("deep_thinking", "1") in ("1", "true", "True") else 0,
            form.get("status", "ok"), form.get("error") or None,
            form.get("case_text_sha256", ""),
            int(form.get("rubric_version", case["rubric_version"])),
            1 if case["owner_id"] == g.runner_user_id else 0,
            self_id_warning, now_iso(),
        ),
    )
    db.commit()
    response = {"stored": True}
    if self_id_warning:
        response["self_id_warning"] = self_id_warning
    return jsonify(response)


@bp.route("/api/runner/jobs/<int:job_id>/status", methods=("POST",))
@token_required
def set_job_status(job_id):
    db = get_db()
    job = db.execute(
        "SELECT * FROM run_jobs WHERE id = ? AND assigned_to = ?",
        (job_id, g.runner_user_id),
    ).fetchone()
    if job is None:
        return jsonify({"error": "unknown run job for this token"}), 404
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    if status not in ("done", "failed", "open"):
        return jsonify({"error": "status must be done, failed, or open"}), 400
    db.execute(
        "UPDATE run_jobs SET status = ?, status_note = ?, updated_at = ? "
        "WHERE id = ?",
        (status, body.get("note", ""), now_iso(), job_id),
    )
    db.commit()
    return jsonify({"stored": True})
