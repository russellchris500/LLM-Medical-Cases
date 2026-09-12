"""The signed-in home page: a to-do list, not a menu.

Physicians land on "what needs me right now" - counts with one big
button each - instead of hunting through pages. Also serves the
plain-language Help page.
"""

from flask import Blueprint, g, redirect, render_template, url_for

from .auth import is_grader, login_required
from .db import get_db
from .grading import grading_pending_count, grading_todo_count

bp = Blueprint("home", __name__)


def dashboard_counts(db):
    """Everything the home page shows, cheapest queries first."""
    counts = {
        "to_grade": 0, "to_finish": 0, "reruns": 0, "open_jobs": 0,
        "flags": 0, "cases": 0, "judge_runs": 0,
    }
    user_id = g.user["id"]
    if is_grader(g.user):
        counts["to_grade"] = grading_todo_count(db, user_id)
        counts["to_finish"] = grading_pending_count(db, user_id)
        counts["cases"] = db.execute(
            "SELECT COUNT(*) AS n FROM cases WHERE owner_id = ? AND deleted = 0",
            (user_id,),
        ).fetchone()["n"]
    if g.user["role"] == "pi":
        counts["reruns"] = db.execute(
            "SELECT COUNT(*) AS n FROM answers WHERE status = 'discarded'"
        ).fetchone()["n"]
        counts["flags"] = db.execute(
            "SELECT COUNT(*) AS n FROM rubric_flags WHERE status = 'open'"
        ).fetchone()["n"]
    else:
        counts["reruns"] = db.execute(
            "SELECT COUNT(*) AS n FROM answers JOIN cases "
            "ON cases.id = answers.case_id "
            "WHERE answers.status = 'discarded' AND cases.owner_id = ?",
            (user_id,),
        ).fetchone()["n"]
        counts["flags"] = db.execute(
            "SELECT COUNT(*) AS n FROM rubric_flags WHERE status = 'open' "
            "AND case_id IN (SELECT id FROM cases WHERE owner_id = ?)",
            (user_id,),
        ).fetchone()["n"]
    counts["judge_runs"] = db.execute(
        "SELECT COUNT(*) AS n FROM judge_runs WHERE assigned_to = ? "
        "AND status = 'open'",
        (user_id,),
    ).fetchone()["n"]
    counts["open_jobs"] = db.execute(
        "SELECT COUNT(*) AS n FROM run_jobs WHERE assigned_to = ? "
        "AND status = 'open'",
        (user_id,),
    ).fetchone()["n"]
    return counts


@bp.app_context_processor
def inject_nav_counts():
    """The 'Grade (n)' badge in the navigation - the one number every
    grader wants to see from any page."""
    user = g.get("user")
    if user is None or not is_grader(user):
        return {"nav_grade_count": 0}
    return {"nav_grade_count": grading_todo_count(get_db(), user["id"])}


@bp.route("/home")
@login_required
def dashboard():
    return render_template("home.html", counts=dashboard_counts(get_db()))


@bp.route("/help")
@login_required
def help_page():
    return render_template("help.html")
