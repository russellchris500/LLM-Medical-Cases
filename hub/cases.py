"""Case authoring for graders: create and edit cases and rubrics.

Same rules as the desktop Case Editor, enforced centrally:
- case IDs are GNNN-CCC (grader number + sequence) and are never reused,
  even after deletion (max_assigned_case_number only goes up);
- a case needs non-empty text and at least one non-empty rubric item;
- ANY change to the rubric's content bumps rubric_version and appends a
  history entry (old items, new items, when) - the audit trail that later
  phases use to invalidate exactly the grades a change affects.
"""

import json

from flask import (
    Blueprint, abort, flash, g, redirect, render_template, request, url_for
)

from case_editor import now_iso
from .auth import login_required
from .db import get_db

bp = Blueprint("cases", __name__)


def case_id_for(grader_number, case_number):
    return "G{:03d}-{:03d}".format(grader_number, case_number)


def clean_case_input(case_text, rubric_lines):
    case_text = (case_text or "").rstrip()
    rubric = [line.strip() for line in (rubric_lines or "").splitlines() if line.strip()]
    if not case_text.strip():
        return None, None, "The case text must not be empty."
    if not rubric:
        return None, None, "The rubric needs at least one item (one per line)."
    return case_text, rubric, None


def load_case(db, case_id, owner_id=None):
    row = db.execute(
        "SELECT * FROM cases WHERE id = ? AND deleted = 0", (case_id,)
    ).fetchone()
    if row is None:
        return None
    if owner_id is not None and row["owner_id"] != owner_id:
        return None
    return row


def create_case(db, owner, case_text, rubric):
    """Allocate the next never-reused case number for this grader."""
    number = owner["max_assigned_case_number"] + 1
    case_id = case_id_for(owner["grader_number"], number)
    db.execute(
        "INSERT INTO cases (id, owner_id, case_number, case_text, rubric) "
        "VALUES (?, ?, ?, ?, ?)",
        (case_id, owner["id"], number, case_text, json.dumps(rubric)),
    )
    db.execute(
        "UPDATE users SET max_assigned_case_number = ? WHERE id = ?",
        (number, owner["id"]),
    )
    db.commit()
    return case_id


def update_case(db, row, case_text, rubric):
    """Apply an edit; rubric content changes bump rubric_version and are
    recorded in rubric_history."""
    old_rubric = json.loads(row["rubric"])
    version = row["rubric_version"]
    history = json.loads(row["rubric_history"])
    if rubric != old_rubric:
        history.append({
            "from_version": version,
            "to_version": version + 1,
            "old_rubric": old_rubric,
            "new_rubric": list(rubric),
            "ops": None,  # structured ops arrive with the grading phase
            "edited_at": now_iso(),
        })
        version += 1
    db.execute(
        "UPDATE cases SET case_text = ?, rubric = ?, rubric_version = ?, "
        "rubric_history = ?, updated_at = ? WHERE id = ?",
        (case_text, json.dumps(rubric), version, json.dumps(history),
         now_iso(), row["id"]),
    )
    db.commit()


@bp.route("/cases")
@login_required
def my_cases():
    db = get_db()
    if g.user["role"] == "pi":
        rows = db.execute(
            "SELECT cases.*, users.name AS owner_name FROM cases "
            "JOIN users ON users.id = cases.owner_id "
            "WHERE cases.deleted = 0 ORDER BY cases.id"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT cases.*, ? AS owner_name FROM cases "
            "WHERE owner_id = ? AND deleted = 0 ORDER BY case_number",
            (g.user["name"], g.user["id"]),
        ).fetchall()
    cases = []
    for row in rows:
        entry = dict(row)
        entry["rubric_items"] = len(json.loads(row["rubric"]))
        cases.append(entry)
    return render_template("cases.html", cases=cases)


@bp.route("/cases/new", methods=("GET", "POST"))
@login_required
def new_case():
    if g.user["role"] != "grader":
        return ("Only graders author cases (the PI coordinates).", 403)
    error = None
    if request.method == "POST":
        case_text, rubric, error = clean_case_input(
            request.form.get("case_text"), request.form.get("rubric")
        )
        if error is None:
            case_id = create_case(get_db(), g.user, case_text, rubric)
            flash("Saved case {}.".format(case_id))
            return redirect(url_for("cases.my_cases"))
    return render_template(
        "case_edit.html", case=None, error=error,
        form_text=request.form.get("case_text", ""),
        form_rubric=request.form.get("rubric", ""),
    )


@bp.route("/cases/<case_id>", methods=("GET", "POST"))
@login_required
def edit_case(case_id):
    db = get_db()
    owner_id = None if g.user["role"] == "pi" else g.user["id"]
    row = load_case(db, case_id, owner_id)
    if row is None:
        abort(404)
    read_only = g.user["role"] == "pi" and row["owner_id"] != g.user["id"]
    error = None
    if request.method == "POST" and not read_only:
        case_text, rubric, error = clean_case_input(
            request.form.get("case_text"), request.form.get("rubric")
        )
        if error is None:
            old_version = row["rubric_version"]
            update_case(db, row, case_text, rubric)
            row = load_case(db, case_id, owner_id)
            if row["rubric_version"] != old_version:
                flash(
                    "Saved case {} - the rubric changed, so it is now rubric "
                    "version {}.".format(case_id, row["rubric_version"])
                )
            else:
                flash("Saved case {}.".format(case_id))
            return redirect(url_for("cases.my_cases"))
    return render_template(
        "case_edit.html", case=row, error=error, read_only=read_only,
        form_text=request.form.get("case_text") or row["case_text"],
        form_rubric=request.form.get("rubric")
        or "\n".join(json.loads(row["rubric"])),
    )


@bp.route("/cases/<case_id>/delete", methods=("POST",))
@login_required
def delete_case(case_id):
    db = get_db()
    row = load_case(db, case_id, g.user["id"] if g.user["role"] == "grader" else None)
    if row is None:
        abort(404)
    db.execute("UPDATE cases SET deleted = 1, updated_at = ? WHERE id = ?",
               (now_iso(), case_id))
    db.commit()
    flash("Deleted case {} (its number is never reused).".format(case_id))
    return redirect(url_for("cases.my_cases"))
