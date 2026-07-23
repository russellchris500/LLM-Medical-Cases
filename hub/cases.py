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
import re

from flask import (
    Blueprint, abort, flash, g, redirect, render_template, request, url_for
)

from case_editor import now_iso
from score_answers import compute_score
from .auth import grant_grader_number, is_grader, login_required
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


# Leading list markers ("1.", "2)", "3]", "-", "*", bullets) are layout,
# not substance: renumbering after a deletion must not count as rewording.
_ENUM_PREFIX = re.compile(r"^\s*(?:\d+\s*[.)\]:]?|[-*•])\s*")


def normalize_item(text):
    """The substance of a rubric item: enumeration stripped, whitespace
    collapsed, case-insensitive."""
    return " ".join(_ENUM_PREFIX.sub("", text).split()).lower()


def match_items(old, new):
    """Match each NEW item to an OLD item with the same substance.

    Returns (mapping, removed): mapping[i] is the old index whose content
    the i-th new item carries (None for genuinely new/reworded items);
    removed lists old indexes no new item claims. Matching consumes old
    items in order, so duplicates pair up sanely."""
    old_norm = [normalize_item(item) for item in old]
    used = set()
    mapping = []
    for item in new:
        norm = normalize_item(item)
        found = None
        for old_index, old_text in enumerate(old_norm):
            if old_index not in used and old_text == norm:
                found = old_index
                break
        if found is not None:
            used.add(found)
        mapping.append(found)
    removed = [index for index in range(len(old)) if index not in used]
    return mapping, removed


def rubric_ops(old, new):
    """What happened to each old item, computed by sequence diff over the
    NORMALIZED items (so renumbering reads as 'kept'): kept, reworded
    (position-paired replacement), removed, or brand-new. Recorded in the
    audit history; grade decisions use match_items instead."""
    old_norm = [normalize_item(item) for item in old]
    new_norm = [normalize_item(item) for item in new]
    ops = {"removed": [], "reworded": [], "added": []}
    matcher = difflib.SequenceMatcher(None, old_norm, new_norm, autojunk=False)
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


def invalidate_grades(db, case_id, old_rubric, new_rubric, new_version):
    """The blast-radius rules, applied instantly to every grader's grades
    on this case, decided by item CONTENT rather than position - so
    deletions, renumberings, and reorderings never throw away a grade
    they do not have to:

    - every kept item inherits its judgment (results remapped by content,
      score recomputed);
    - grades that missed ONLY removed items are set aside for re-grading
      (their risk/approach questions were never asked);
    - any genuinely new or reworded item sets every older grade aside -
      a fresh judgment is needed.
    Returns (requeued, carried, has_new_content)."""
    mapping, removed = match_items(old_rubric, new_rubric)
    has_new_content = any(index is None for index in mapping)
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
        carry = not has_new_content and len(results) == len(old_rubric)
        if carry:
            missed_removed = any(
                results[index] is False for index in removed
            )
            covered_kept = all(results[index] for index in mapping)
            if missed_removed and covered_kept:
                # The score would now hinge on the risk/approach
                # questions that were never asked.
                carry = False
        if carry:
            new_results = [bool(results[index]) for index in mapping]
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
    return requeued, carried, has_new_content


def update_case(db, row, case_text, rubric):
    """Apply an edit; rubric content changes bump rubric_version, are
    recorded in rubric_history, and instantly re-queue exactly the grades
    the change affects. Returns an info dict for the page's message."""
    old_rubric = json.loads(row["rubric"])
    version = row["rubric_version"]
    history = json.loads(row["rubric_history"])
    info = {"rubric_changed": False, "new_version": version,
            "requeued": 0, "carried": 0, "new_items": False}
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
        info["requeued"], info["carried"], info["new_items"] = (
            invalidate_grades(db, row["id"], old_rubric, rubric, version + 1)
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
    if not is_grader(g.user):
        if g.user["role"] != "pi":
            return ("Only graders author cases.", 403)
        # The PI wears the grader hat too - opt in on first use.
        db = get_db()
        number = grant_grader_number(db, g.user["id"])
        g.user = db.execute(
            "SELECT * FROM users WHERE id = ?", (g.user["id"],)
        ).fetchone()
        flash(
            "You are now also grader {} - your cases will get IDs like "
            "G{:03d}-001, and the Grade and My results pages are open to "
            "you.".format(number, number)
        )
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
                    if info["new_items"]:
                        message += (" (An item was added or reworded, so a "
                                    "fresh judgment is needed on every "
                                    "answer.)")
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
    # Only a case's owner may delete it - including a PI, only for their own.
    row = load_case(db, case_id, g.user["id"])
    if row is None:
        abort(404)
    db.execute("UPDATE cases SET deleted = 1, updated_at = ? WHERE id = ?",
               (now_iso(), case_id))
    db.commit()
    flash("Deleted case {} (its number is never reused).".format(case_id))
    return redirect(url_for("cases.my_cases"))
