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
    audit history; grade decisions use classify_change instead."""
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


def classify_change(old, new):
    """What a rubric save did to each line, decided by CONTENT first.

    Returns (mapping, changed, added, deleted):
    - mapping[j] = old index whose substance the j-th new item carries
      (so reorderings and renumberings are 'kept'), None otherwise;
    - changed = {new_index: old_index} for leftover lines that sit in
      the same replaced block - a rewording of one item;
    - added = new lines with no old counterpart at all;
    - deleted = old lines nothing in the new rubric accounts for."""
    mapping, removed = match_items(old, new)
    old_norm = [normalize_item(item) for item in old]
    new_norm = [normalize_item(item) for item in new]
    changed = {}
    matcher = difflib.SequenceMatcher(None, old_norm, new_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        for step in range(min(i2 - i1, j2 - j1)):
            old_index, new_index = i1 + step, j1 + step
            if mapping[new_index] is None and old_index in removed:
                changed[new_index] = old_index
    added = [index for index, source in enumerate(mapping)
             if source is None and index not in changed]
    deleted = [index for index in removed if index not in changed.values()]
    return mapping, changed, added, deleted


def partial_score(results, risk, poor):
    """The usual 0/1/2 score, or None while anything it depends on is
    still unanswered - the mark of a pending grade awaiting its grader."""
    if any(result is None for result in results):
        return None
    if not all(results):
        return 0
    if risk is None:
        return None
    if risk:
        return 0
    if poor is None:
        return None
    return 1 if poor else 2


def invalidate_grades(db, case_id, old_rubric, new_rubric, new_version,
                      reset_items=None):
    """Carry every judgment a rubric edit does not touch, applied
    instantly to every grader's grades on this case:

    - kept items (including reordered/renumbered ones) keep their
      judgment;
    - a deleted item's judgment vanishes with it;
    - an added item is left unanswered - the grader judges just that;
    - a changed item keeps its judgment unless the editor chose to
      reset it (reset_items = new-rubric indexes to re-judge);
    - the score is recomputed; if anything is now unanswered (a new
      item, a reset item, or the risk/approach questions that were
      never asked) the grade stays active with a NULL score until the
      grader completes it.
    Returns (complete, pending, superseded) counts."""
    reset_items = reset_items or set()
    mapping, changed, _added, _deleted = classify_change(old_rubric, new_rubric)
    complete = pending = superseded = 0
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
        if len(results) != len(old_rubric):
            # A grade that predates the recorded rubric - unmappable.
            db.execute(
                "UPDATE grades SET superseded = 1, superseded_reason = ?, "
                "updated_at = ? WHERE id = ?",
                ("rubric changed to version {}".format(new_version),
                 now_iso(), grade["id"]),
            )
            superseded += 1
            continue
        new_results = []
        for index in range(len(new_rubric)):
            source = mapping[index]
            if source is None and index in changed and index not in reset_items:
                source = changed[index]
            if source is None:
                new_results.append(None)
            else:
                value = results[source]
                new_results.append(None if value is None else bool(value))
        score = partial_score(new_results, grade["unnecessary_risk"],
                              grade["poor_approach"])
        db.execute(
            "UPDATE grades SET rubric_results = ?, score = ?, "
            "rubric_version = ?, updated_at = ? WHERE id = ?",
            (json.dumps(new_results), score, new_version, now_iso(),
             grade["id"]),
        )
        if score is None:
            pending += 1
        else:
            complete += 1
    return complete, pending, superseded


def discard_all_answers(db, case_id, editor_id):
    """The case text changed and the editor chose to drop the answers
    (they answered the OLD question). Same soft-discard as grading's
    discard button: each answer shows on the Run jobs page as needing a
    re-run, and a fresh upload revives its slot under the same letter."""
    rows = db.execute(
        "SELECT id FROM answers WHERE case_id = ? AND status != 'discarded'",
        (case_id,),
    ).fetchall()
    for row in rows:
        db.execute(
            "UPDATE answers SET status = 'discarded', discarded_by = ?, "
            "discarded_reason = 'case text edited' WHERE id = ?",
            (editor_id, row["id"]),
        )
        db.execute(
            "UPDATE grades SET superseded = 1, "
            "superseded_reason = 'answer discarded: case text edited', "
            "updated_at = ? WHERE answer_id = ? AND superseded = 0",
            (now_iso(), row["id"]),
        )
    return len(rows)


def update_case(db, row, case_text, rubric, decisions, editor_id):
    """Apply an edit; rubric content changes bump rubric_version, are
    recorded in rubric_history, and instantly remap every grader's
    grades per the blast-radius rules. decisions carries the editor's
    confirmed choices: answers_action ('keep'/'delete'/None) for a
    case-text change with existing answers, and reset_items (set of
    new-rubric indexes) for changed lines whose judgments are dropped.
    Returns an info dict for the page's message."""
    old_rubric = json.loads(row["rubric"])
    version = row["rubric_version"]
    history = json.loads(row["rubric_history"])
    info = {"rubric_changed": False, "new_version": version,
            "complete": 0, "pending": 0, "superseded": 0,
            "answers_discarded": 0}
    if decisions.get("answers_action") == "delete":
        info["answers_discarded"] = discard_all_answers(
            db, row["id"], editor_id
        )
    if rubric != old_rubric:
        ops = rubric_ops(old_rubric, rubric)
        history.append({
            "from_version": version,
            "to_version": version + 1,
            "old_rubric": old_rubric,
            "new_rubric": list(rubric),
            "ops": ops,
            "reset_items": sorted(decisions.get("reset_items") or ()),
            "edited_at": now_iso(),
        })
        info["complete"], info["pending"], info["superseded"] = (
            invalidate_grades(db, row["id"], old_rubric, rubric, version + 1,
                              decisions.get("reset_items"))
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


def edit_summary(case_id, info):
    """The flash message after a confirmed edit."""
    if info["rubric_changed"]:
        message = ("Saved case {} - the rubric changed, so it is now rubric "
                   "version {}.".format(case_id, info["new_version"]))
        if info["complete"]:
            message += (" {} grade(s) carried over in full."
                        .format(info["complete"]))
        if info["pending"]:
            message += (" {} grade(s) carried over but need finishing - "
                        "each grader answers only the new or changed item "
                        "(or the risk/approach questions)."
                        .format(info["pending"]))
        if info["superseded"]:
            message += (" {} grade(s) were set aside for a full re-grade."
                        .format(info["superseded"]))
    else:
        message = "Saved case {}.".format(case_id)
    if info["answers_discarded"]:
        message += (" {} answer(s) were discarded because the question "
                    "changed - re-run them from the Run jobs page."
                    .format(info["answers_discarded"]))
    return message


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
    confirm_questions = None
    confirm_hidden = {}
    if request.method == "POST" and not read_only:
        case_text, rubric, error = clean_case_input(
            request.form.get("case_text"), request.form.get("rubric")
        )
        if error is None:
            old_rubric = json.loads(row["rubric"])
            text_changed = case_text != row["case_text"]
            _mapping, changed, _added, _deleted = classify_change(
                old_rubric, rubric
            )
            answers_n = db.execute(
                "SELECT COUNT(*) AS n FROM answers WHERE case_id = ? "
                "AND status != 'discarded'", (case_id,)
            ).fetchone()["n"]
            grades_n = db.execute(
                "SELECT COUNT(*) AS n FROM grades JOIN grading_assignments ga "
                "ON ga.id = grades.assignment_id "
                "WHERE ga.case_id = ? AND grades.superseded = 0",
                (case_id,),
            ).fetchone()["n"]
            # Decisions the editor must confirm before the save applies.
            questions = []
            hidden = {}
            answers_action = request.form.get("answers_action")
            ask_answers = text_changed and answers_n > 0
            if ask_answers:
                if answers_action in ("keep", "delete"):
                    hidden["answers_action"] = answers_action
                else:
                    questions.append({"kind": "answers", "count": answers_n})
            reset_items = set()
            if rubric != old_rubric and grades_n:
                for new_index in sorted(changed):
                    choice = request.form.get("changed_{}".format(new_index))
                    if choice in ("keep", "reset"):
                        hidden["changed_{}".format(new_index)] = choice
                        if choice == "reset":
                            reset_items.add(new_index)
                    else:
                        questions.append({
                            "kind": "changed", "index": new_index,
                            "old": old_rubric[changed[new_index]],
                            "new": rubric[new_index],
                        })
            if questions:
                confirm_questions = questions
                confirm_hidden = hidden
            else:
                decisions = {
                    "answers_action": answers_action if ask_answers else None,
                    "reset_items": reset_items,
                }
                info = update_case(db, row, case_text, rubric, decisions,
                                   g.user["id"])
                flash(edit_summary(case_id, info))
                return redirect(url_for("cases.my_cases"))
    has_answers = db.execute(
        "SELECT COUNT(*) AS n FROM answers WHERE case_id = ? "
        "AND status IN ('ok', 'ok_manual')", (case_id,)
    ).fetchone()["n"] > 0
    return render_template(
        "case_edit.html", case=row, error=error, read_only=read_only,
        confirm_questions=confirm_questions, confirm_hidden=confirm_hidden,
        has_answers=has_answers,
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
