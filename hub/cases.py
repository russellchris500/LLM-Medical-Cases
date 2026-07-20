"""Case authoring for graders: create and edit cases and rubrics.

Same rules as the desktop Case Editor, enforced centrally:
- case IDs are GNNN-CCC (grader number + sequence) and are never reused,
  even after deletion (max_assigned_case_number only goes up);
- a case needs non-empty text and at least one non-empty rubric item;
- ANY change to the rubric's content bumps rubric_version and appends a
  history entry (old items, new items, when) - the audit trail that later
  phases use to invalidate exactly the grades a change affects.
"""

import difflib
import json

from flask import (
    Blueprint, abort, flash, g, redirect, render_template, request, url_for
)

from case_editor import now_iso
from score_answers import compute_score, grade_survives_removal
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


def rubric_ops(old, new):
    """What happened to each old item, computed by sequence diff: kept,
    reworded (position-paired replacement), removed, or brand-new."""
    ops = {"removed": [], "reworded": [], "added": []}
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            ops["removed"].extend(range(i1, i2))
        elif tag == "insert":
            ops["added"].extend(range(j1, j2))
        elif tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            ops["reworded"].extend(range(i1, i1 + paired))
            ops["removed"].extend(range(i1 + paired, i2))
            ops["added"].extend(range(j1 + paired, j2))
    return ops


def invalidate_grades(db, case_id, ops, from_version, new_version):
    """The blast-radius rules, applied instantly to every grader's grades
    on this case (no update files, no emails):

    - pure removals: grades that covered the removed item(s), or that
      missed some KEPT item anyway, are carried over losslessly (results
      remapped, score recomputed); only grades that missed ONLY removed
      items are set aside for re-grading (their risk/approach questions
      were never asked).
    - anything added or reworded: every older grade on the case is set
      aside for re-grading.
    Returns (requeued, carried)."""
    clean_removal = not (ops["added"] or ops["reworded"])
    removed = ops["removed"]
    requeued = carried = 0
    rows = db.execute(
        "SELECT grades.* FROM grades JOIN grading_assignments ga "
        "ON ga.id = grades.assignment_id "
        "WHERE ga.case_id = ? AND grades.superseded = 0",
        (case_id,),
    ).fetchall()
    for grade in rows:
        if grade["rubric_version"] >= new_version:
            continue
        results = json.loads(grade["rubric_results"])
        if (
            clean_removal
            and grade["rubric_version"] == from_version
            and grade_survives_removal({"rubric_results": results}, removed)
        ):
            new_results = [
                r for i, r in enumerate(results) if i not in removed
            ]
            score = compute_score(
                new_results, grade["unnecessary_risk"], grade["poor_approach"]
            )
            db.execute(
                "UPDATE grades SET rubric_results = ?, score = ?, "
                "rubric_version = ?, updated_at = ? WHERE id = ?",
                (json.dumps(new_results), score, new_version, now_iso(),
                 grade["id"]),
            )
            carried += 1
        else:
            db.execute(
                "UPDATE grades SET superseded = 1, superseded_reason = ?, "
                "updated_at = ? WHERE id = ?",
                ("rubric changed to version {}".format(new_version), now_iso(),
                 grade["id"]),
            )
            requeued += 1
    return requeued, carried


def update_case(db, row, case_text, rubric):
    """Apply an edit; rubric content changes bump rubric_version, are
    recorded in rubric_history, and instantly re-queue exactly the grades
    the change affects. Returns an info dict for the page's message."""
    old_rubric = json.loads(row["rubric"])
    version = row["rubric_version"]
    history = json.loads(row["rubric_history"])
    info = {"rubric_changed": False, "new_version": version,
            "requeued": 0, "carried": 0}
    if rubric != old_rubric:
        ops = rubric_ops(old_rubric, rubric)
        history.append({
            "from_version": version,
            "to_version": version + 1,
            "old_rubric": old_rubric,
            "new_rubric": list(rubric),
            "ops": ops,
            "edited_at": now_iso(),
        })
        info["requeued"], info["carried"] = invalidate_grades(
            db, row["id"], ops, version, version + 1
        )
        version += 1
        info.update(rubric_changed=True, new_version=version)
    db.execute(
        "UPDATE cases SET case_text = ?, rubric = ?, rubric_version = ?, "
        "rubric_history = ?, updated_at = ? WHERE id = ?",
        (case_text, json.dumps(rubric), version, json.dumps(history),
         now_iso(), row["id"]),
    )
    db.commit()
    return info


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
            info = update_case(db, row, case_text, rubric)
            if info["rubric_changed"]:
                message = ("Saved case {} - the rubric changed, so it is now "
                           "rubric version {}.".format(case_id, info["new_version"]))
                if info["requeued"]:
                    message += (" {} grade(s) affected by the change were set "
                                "aside and will come back for re-grading."
                                .format(info["requeued"]))
                if info["carried"]:
                    message += (" {} grade(s) were carried over unchanged."
                                .format(info["carried"]))
                flash(message)
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
