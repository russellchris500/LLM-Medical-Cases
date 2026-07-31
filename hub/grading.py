"""Blinded grading in the browser.

Blinding: for every (grader, case) assignment the hub shuffles the
case's answers into letters server-side (SystemRandom) and never reveals
the mapping. Graders see the answer TEXT rendered from markdown - never
the captured site HTML, whose markup would identify the model - and
images stream through opaque URLs that hide the stored file names.
Where the same model variant was run both by the case owner and by the
PI, the PI-run answer is preferred (it is the truly blind one).

Scoring is the same 0/1/2 rule as always (compute_score); the risk and
approach questions only apply once every rubric item is covered.
"""

import json
import os
import random
import string

from flask import (
    Blueprint, abort, current_app, flash, g, redirect, render_template,
    request, send_file, url_for
)

from case_editor import now_iso
from eval_common import markdown_to_html
from score_answers import compute_score
from .auth import is_grader, login_required, pi_required
from .db import get_db

bp = Blueprint("grading", __name__)

LABELS = string.ascii_uppercase


# ---------- assignments and blinding ----------


def ensure_own_assignments(db, grader_id):
    """A grader automatically grades their own cases that have answers."""
    rows = db.execute(
        "SELECT DISTINCT cases.id FROM cases "
        "JOIN answers ON answers.case_id = cases.id AND answers.status "
        "IN ('ok', 'ok_manual') "
        "WHERE cases.owner_id = ? AND cases.deleted = 0",
        (grader_id,),
    ).fetchall()
    for row in rows:
        db.execute(
            "INSERT OR IGNORE INTO grading_assignments (grader_id, case_id, kind) "
            "VALUES (?, ?, 'own')",
            (grader_id, row["id"]),
        )
    db.commit()


def gradable_answers(db, case_id):
    """One answer per model variant: prefer answers NOT run by the case
    owner (truly blind), then the newest."""
    rows = db.execute(
        "SELECT * FROM answers WHERE case_id = ? AND status IN ('ok', 'ok_manual') "
        "ORDER BY variant_id, run_by_owner ASC, created_at DESC, id DESC",
        (case_id,),
    ).fetchall()
    chosen = {}
    for row in rows:
        chosen.setdefault(row["variant_id"], row)
    return list(chosen.values())


def ensure_blind_labels(db, assignment):
    """Assign shuffled letters to this case's answers, once; answers that
    arrive later get the next letters (shuffled among themselves)."""
    answers = gradable_answers(db, assignment["case_id"])
    existing = db.execute(
        "SELECT * FROM blind_labels WHERE assignment_id = ?", (assignment["id"],)
    ).fetchall()
    labeled_ids = {row["answer_id"] for row in existing}
    used_labels = {row["label"] for row in existing}
    new_answers = [a for a in answers if a["id"] not in labeled_ids]
    if new_answers:
        rng = random.SystemRandom()
        rng.shuffle(new_answers)
        free = [letter for letter in LABELS if letter not in used_labels]
        for answer, label in zip(new_answers, free):
            db.execute(
                "INSERT INTO blind_labels (assignment_id, label, answer_id) "
                "VALUES (?, ?, ?)",
                (assignment["id"], label, answer["id"]),
            )
        db.commit()
    return db.execute(
        "SELECT blind_labels.label, answers.* FROM blind_labels "
        "JOIN answers ON answers.id = blind_labels.answer_id "
        "WHERE assignment_id = ? AND answers.status IN ('ok', 'ok_manual') "
        "ORDER BY blind_labels.label",
        (assignment["id"],),
    ).fetchall()


def grading_todo_count(db, grader_id):
    """How many answers still need this grader (ungraded or pending),
    WITHOUT creating assignments or labels - safe to call on any page."""
    case_ids = {
        row["id"]
        for row in db.execute(
            "SELECT DISTINCT cases.id FROM cases JOIN answers "
            "ON answers.case_id = cases.id "
            "AND answers.status IN ('ok', 'ok_manual') "
            "WHERE cases.owner_id = ? AND cases.deleted = 0",
            (grader_id,),
        )
    }
    case_ids |= {
        row["case_id"]
        for row in db.execute(
            "SELECT ga.case_id FROM grading_assignments ga "
            "JOIN cases ON cases.id = ga.case_id "
            "WHERE ga.grader_id = ? AND cases.deleted = 0",
            (grader_id,),
        )
    }
    todo = 0
    for case_id in case_ids:
        answers = gradable_answers(db, case_id)
        assignment = db.execute(
            "SELECT id FROM grading_assignments "
            "WHERE grader_id = ? AND case_id = ?",
            (grader_id, case_id),
        ).fetchone()
        graded = 0
        if assignment is not None:
            for answer in answers:
                grade = active_grade(db, assignment["id"], answer["id"])
                if grade is not None and grade["score"] is not None:
                    graded += 1
        todo += len(answers) - graded
    return todo


def grading_pending_count(db, grader_id):
    """Grades left half-done by a rubric edit (score NULL) - shown
    separately so the grader knows these are quick finishes."""
    return db.execute(
        "SELECT COUNT(*) AS n FROM grades "
        "JOIN grading_assignments ga ON ga.id = grades.assignment_id "
        "JOIN answers ON answers.id = grades.answer_id "
        "WHERE ga.grader_id = ? AND grades.superseded = 0 "
        "AND grades.score IS NULL "
        "AND answers.status IN ('ok', 'ok_manual')",
        (grader_id,),
    ).fetchone()["n"]


def my_assignment(db, case_id):
    row = db.execute(
        "SELECT * FROM grading_assignments WHERE grader_id = ? AND case_id = ?",
        (g.user["id"], case_id),
    ).fetchone()
    return row


def active_grade(db, assignment_id, answer_id):
    return db.execute(
        "SELECT * FROM grades WHERE assignment_id = ? AND answer_id = ? "
        "AND superseded = 0",
        (assignment_id, answer_id),
    ).fetchone()


# ---------- pages ----------


@bp.route("/grade")
@login_required
def queue():
    if not is_grader(g.user):
        # A PI who has not opted into grading yet.
        return redirect(url_for("grading.assignments_page"))
    db = get_db()
    ensure_own_assignments(db, g.user["id"])
    assignments = db.execute(
        "SELECT grading_assignments.*, cases.case_text, cases.rubric_version "
        "FROM grading_assignments JOIN cases ON cases.id = grading_assignments.case_id "
        "WHERE grader_id = ? AND cases.deleted = 0 ORDER BY cases.id",
        (g.user["id"],),
    ).fetchall()
    queue_rows = []
    for assignment in assignments:
        labeled = ensure_blind_labels(db, assignment)
        graded = 0
        for answer in labeled:
            grade = active_grade(db, assignment["id"], answer["id"])
            # A pending grade (score NULL after a rubric edit) still
            # needs the grader, so it does not count as done.
            if grade is not None and grade["score"] is not None:
                graded += 1
        queue_rows.append({
            "assignment": assignment,
            "answers": len(labeled),
            "graded": graded,
        })
    return render_template("grade_queue.html", rows=queue_rows)


@bp.route("/grade/next")
@login_required
def next_answer():
    """One click from anywhere to the next answer that needs this
    grader - across ALL their cases."""
    if not is_grader(g.user):
        return redirect(url_for("grading.queue"))
    db = get_db()
    ensure_own_assignments(db, g.user["id"])
    assignments = db.execute(
        "SELECT grading_assignments.* FROM grading_assignments "
        "JOIN cases ON cases.id = grading_assignments.case_id "
        "WHERE grader_id = ? AND cases.deleted = 0 ORDER BY cases.id",
        (g.user["id"],),
    ).fetchall()
    for assignment in assignments:
        for answer in ensure_blind_labels(db, assignment):
            grade = active_grade(db, assignment["id"], answer["id"])
            if grade is None or grade["score"] is None:
                return redirect(url_for(
                    "grading.answer_page",
                    case_id=assignment["case_id"], label=answer["label"],
                ))
    flash("You are all caught up - nothing left to grade right now.")
    return redirect(url_for("grading.queue"))


@bp.route("/grade/<case_id>")
@login_required
def case_page(case_id):
    db = get_db()
    assignment = my_assignment(db, case_id)
    if assignment is None and is_grader(g.user):
        # The builder's "Grade this case" shortcut may arrive before the
        # queue page ever created the owner's assignment - create it now.
        owned = db.execute(
            "SELECT 1 FROM cases WHERE id = ? AND owner_id = ? AND deleted = 0",
            (case_id, g.user["id"]),
        ).fetchone()
        if owned is not None:
            ensure_own_assignments(db, g.user["id"])
            assignment = my_assignment(db, case_id)
    if assignment is None:
        abort(404)
    case = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    labeled = ensure_blind_labels(db, assignment)
    entries = []
    for answer in labeled:
        grade = active_grade(db, assignment["id"], answer["id"])
        entries.append({
            "label": answer["label"],
            "graded": grade is not None and grade["score"] is not None,
            "pending": grade is not None and grade["score"] is None,
            "score": grade["score"] if grade else None,
        })
    return render_template(
        "grade_case.html", case=case, entries=entries,
        rubric=json.loads(case["rubric"]),
        can_edit=case["owner_id"] == g.user["id"] or g.user["role"] == "pi",
    )


@bp.route("/grade/<case_id>/<label>", methods=("GET", "POST"))
@login_required
def answer_page(case_id, label):
    db = get_db()
    assignment = my_assignment(db, case_id)
    if assignment is None:
        abort(404)
    case = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    answer = db.execute(
        "SELECT blind_labels.label, answers.* FROM blind_labels "
        "JOIN answers ON answers.id = blind_labels.answer_id "
        "WHERE assignment_id = ? AND label = ? "
        "AND answers.status IN ('ok', 'ok_manual')",
        (assignment["id"], label.upper()),
    ).fetchone()
    if answer is None:
        abort(404)
    rubric = json.loads(case["rubric"])
    previous = active_grade(db, assignment["id"], answer["id"])
    error = None

    if request.method == "POST":
        results = []
        for index in range(len(rubric)):
            value = request.form.get("item_{}".format(index))
            if value not in ("yes", "no"):
                error = "Please answer Covered or Missed for every rubric item."
                break
            results.append(value == "yes")
        if error is None:
            risk = poor = None
            if all(results):
                risk_value = request.form.get("risk")
                if risk_value not in ("yes", "no"):
                    error = "Please answer the unnecessary-risk question."
                elif risk_value == "yes":
                    risk = True
                else:
                    risk = False
                    poor_value = request.form.get("poor")
                    if poor_value not in ("yes", "no"):
                        error = "Please answer the poor-approach question."
                    else:
                        poor = poor_value == "yes"
        if error is None:
            score = compute_score(results, risk, poor)
            comment = request.form.get("comment", "").strip()
            if previous is not None:
                db.execute(
                    "UPDATE grades SET rubric_results = ?, unnecessary_risk = ?, "
                    "poor_approach = ?, score = ?, comment = ?, rubric_version = ?, "
                    "updated_at = ? WHERE id = ?",
                    (json.dumps(results), risk, poor, score, comment,
                     case["rubric_version"], now_iso(), previous["id"]),
                )
            else:
                db.execute(
                    "INSERT INTO grades (assignment_id, answer_id, rubric_results, "
                    "unnecessary_risk, poor_approach, score, comment, "
                    "rubric_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (assignment["id"], answer["id"], json.dumps(results),
                     risk, poor, score, comment, case["rubric_version"]),
                )
            db.commit()
            flash("Saved: answer {} of case {} scored {}.".format(
                answer["label"], case_id, score
            ))
            # Next answer still needing this grader: first within this
            # case, then automatically on to their next case.
            for row in ensure_blind_labels(db, assignment):
                grade = active_grade(db, assignment["id"], row["id"])
                if grade is None or grade["score"] is None:
                    return redirect(url_for(
                        "grading.answer_page", case_id=case_id, label=row["label"]
                    ))
            return redirect(url_for("grading.next_answer"))

    answer_html = markdown_to_html(
        answer["response_text"] or "(no text)", title="Answer " + answer["label"]
    )
    # Strip the outer document; keep just the body for embedding.
    body = answer_html.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    image_count = len(json.loads(answer["image_paths"]))
    form = {
        "results": json.loads(previous["rubric_results"]) if previous else
                   [None] * len(rubric),
        "risk": previous["unnecessary_risk"] if previous else None,
        "poor": previous["poor_approach"] if previous else None,
        "comment": previous["comment"] if previous else "",
    }
    # "Answer 2 of 5" progress within this case.
    labels = [row["label"] for row in ensure_blind_labels(db, assignment)]
    position = labels.index(answer["label"]) + 1 if answer["label"] in labels else 1
    # A grade left half-done by a rubric edit: tell the grader exactly
    # what is still owed.
    pending_note = None
    if previous is not None and previous["score"] is None:
        if any(value is None for value in form["results"]):
            pending_note = ("The rubric changed since you graded this "
                            "answer. Your other judgments were kept - only "
                            "the unanswered item(s) below need you.")
        else:
            pending_note = ("The rubric changed since you graded this "
                            "answer. Every item judgment was kept - only "
                            "the final questions below still need answers.")
    return render_template(
        "grade_answer.html", case=case, answer=answer, rubric=rubric,
        answer_body=body, image_count=image_count, form=form, error=error,
        run_blind=not answer["run_by_owner"],
        position=position, total=len(labels), pending_note=pending_note,
    )


@bp.route("/grade/<case_id>/<label>/image/<int:index>")
@login_required
def answer_image(case_id, label, index):
    db = get_db()
    assignment = my_assignment(db, case_id)
    if assignment is None:
        abort(404)
    answer = db.execute(
        "SELECT answers.* FROM blind_labels "
        "JOIN answers ON answers.id = blind_labels.answer_id "
        "WHERE assignment_id = ? AND label = ? "
        "AND answers.status IN ('ok', 'ok_manual')",
        (assignment["id"], label.upper()),
    ).fetchone()
    if answer is None:
        abort(404)
    paths = json.loads(answer["image_paths"])
    if not 0 <= index < len(paths):
        abort(404)
    full = os.path.join(current_app.config["ANSWER_DIR"], paths[index])
    if not os.path.exists(full):
        abort(404)
    # Opaque download name: the stored name contains the model identity.
    extension = os.path.splitext(full)[1] or ".png"
    return send_file(full, download_name="{}_{}_{}{}".format(
        case_id, label.upper(), index + 1, extension
    ))


@bp.route("/grade/<case_id>/<label>/discard", methods=("POST",))
@login_required
def discard_answer(case_id, label):
    """An incomplete / badly captured answer is taken out of the study:
    it disappears from every grader's pages, all grades on it are set
    aside, and the case owner sees it on the Run jobs page for a re-run.
    A fresh run of the same case and model replaces it in place."""
    db = get_db()
    assignment = my_assignment(db, case_id)
    if assignment is None:
        abort(404)
    answer = db.execute(
        "SELECT answers.* FROM blind_labels "
        "JOIN answers ON answers.id = blind_labels.answer_id "
        "WHERE assignment_id = ? AND label = ? "
        "AND answers.status IN ('ok', 'ok_manual')",
        (assignment["id"], label.upper()),
    ).fetchone()
    if answer is None:
        abort(404)
    reason = request.form.get("reason", "").strip() or "incomplete capture"
    db.execute(
        "UPDATE answers SET status = 'discarded', discarded_by = ?, "
        "discarded_reason = ? WHERE id = ?",
        (g.user["id"], reason, answer["id"]),
    )
    superseded = db.execute(
        "UPDATE grades SET superseded = 1, "
        "superseded_reason = 'answer discarded as incomplete', updated_at = ? "
        "WHERE answer_id = ? AND superseded = 0",
        (now_iso(), answer["id"]),
    ).rowcount
    db.commit()
    message = ("Answer {} of case {} was discarded ({}). Re-run that case "
               "on the Run jobs page - the fresh answer will take the same "
               "letter and come back for grading.".format(
                   label.upper(), case_id, reason))
    if superseded:
        message += " {} existing grade(s) on it were set aside.".format(superseded)
    flash(message)
    return redirect(url_for("grading.case_page", case_id=case_id))


@bp.route("/grade/<case_id>/flag", methods=("POST",))
@login_required
def flag_item(case_id):
    db = get_db()
    assignment = my_assignment(db, case_id)
    if assignment is None:
        abort(404)
    case = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    rubric = json.loads(case["rubric"])
    try:
        index = int(request.form.get("item_index", "-1"))
    except ValueError:
        index = -1
    note = request.form.get("note", "").strip()
    if not 0 <= index < len(rubric) or not note:
        flash("Pick a rubric item and describe the problem.")
    else:
        db.execute(
            "INSERT INTO rubric_flags (case_id, item_index, item_text, note, "
            "flagged_by) VALUES (?, ?, ?, ?, ?)",
            (case_id, index, rubric[index], note, g.user["id"]),
        )
        db.commit()
        owner = db.execute(
            "SELECT owner_id FROM cases WHERE id = ?", (case_id,)
        ).fetchone()["owner_id"]
        flash(
            "Flag saved. Keep grading against the current wording - {} "
            "and the rubric fix (if any) will re-queue exactly the answers "
            "it affects.".format(
                "you own this case, so fix the rubric on its edit page"
                if owner == g.user["id"] else
                "the case's owner sees your flag"
            )
        )
    return redirect(url_for("grading.case_page", case_id=case_id))


# ---------- flags for case owners ----------


@bp.route("/flags", methods=("GET", "POST"))
@login_required
def flags_page():
    db = get_db()
    if request.method == "POST":
        db.execute(
            "UPDATE rubric_flags SET status = 'resolved' WHERE id = ? AND "
            "case_id IN (SELECT id FROM cases WHERE owner_id = ?)",
            (request.form.get("resolve"), g.user["id"]),
        )
        db.commit()
        return redirect(url_for("grading.flags_page"))
    if g.user["role"] == "pi":
        rows = db.execute(
            "SELECT rubric_flags.*, users.name AS flagger FROM rubric_flags "
            "JOIN users ON users.id = rubric_flags.flagged_by "
            "WHERE status = 'open' ORDER BY rubric_flags.id DESC"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT rubric_flags.*, users.name AS flagger FROM rubric_flags "
            "JOIN users ON users.id = rubric_flags.flagged_by "
            "WHERE status = 'open' AND case_id IN "
            "(SELECT id FROM cases WHERE owner_id = ?) "
            "ORDER BY rubric_flags.id DESC",
            (g.user["id"],),
        ).fetchall()
    return render_template("flags.html", flags=rows)


# ---------- PI: cross-grading assignments ----------


@bp.route("/assignments", methods=("GET", "POST"))
@pi_required
def assignments_page():
    db = get_db()
    error = None
    if request.method == "POST":
        grader_id = request.form.get("grader_id")
        case_ids = request.form.getlist("case_id")
        grader = db.execute(
            "SELECT * FROM users WHERE id = ? AND grader_number IS NOT NULL",
            (grader_id,),
        ).fetchone()
        if grader is None or not case_ids:
            error = "Pick a grader and at least one case."
        else:
            added = 0
            for case_id in case_ids:
                case = db.execute(
                    "SELECT * FROM cases WHERE id = ? AND deleted = 0", (case_id,)
                ).fetchone()
                if case is None or case["owner_id"] == grader["id"]:
                    continue  # cross-grading means someone ELSE's case
                cursor = db.execute(
                    "INSERT OR IGNORE INTO grading_assignments "
                    "(grader_id, case_id, kind, assigned_by) "
                    "VALUES (?, ?, 'cross', ?)",
                    (grader["id"], case_id, g.user["id"]),
                )
                added += cursor.rowcount
            db.commit()
            flash("Assigned {} case(s) to {} for cross-grading.".format(
                added, grader["name"]
            ))
            return redirect(url_for("grading.assignments_page"))
    graders = db.execute(
        "SELECT * FROM users WHERE grader_number IS NOT NULL AND disabled = 0 "
        "ORDER BY name COLLATE NOCASE"
    ).fetchall()
    cases = db.execute(
        "SELECT cases.*, users.name AS owner_name FROM cases "
        "JOIN users ON users.id = cases.owner_id WHERE cases.deleted = 0 "
        "ORDER BY cases.id"
    ).fetchall()
    crosses = db.execute(
        "SELECT grading_assignments.*, users.name AS grader_name FROM "
        "grading_assignments JOIN users ON users.id = grading_assignments.grader_id "
        "WHERE kind = 'cross' ORDER BY grading_assignments.id DESC"
    ).fetchall()
    return render_template(
        "assignments.html", graders=graders, cases=cases, crosses=crosses,
        error=error,
    )
