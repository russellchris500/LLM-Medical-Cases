"""Accounts and sessions: PI-seeded, invite-link onboarding, no self-signup.

The first PI account is created from the command line (manage.py). The PI
invites graders from the People page; each invite is a one-time link where
the grader picks their password. Passwords are stored hashed (werkzeug).
"""

import functools
import secrets

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from .db import get_db

bp = Blueprint("auth", __name__)


def is_grader(user):
    """Anyone holding a grader number authors, runs, and grades - the PI
    included, once they opt in (roles are hats, not walls)."""
    return user is not None and user["grader_number"] is not None


@bp.app_context_processor
def inject_user():
    user = g.get("user")
    return {"user": user, "user_is_grader": is_grader(user)}


@bp.before_app_request
def load_user():
    g.user = None
    user_id = session.get("user_id")
    if user_id is not None:
        row = get_db().execute(
            "SELECT * FROM users WHERE id = ? AND disabled = 0", (user_id,)
        ).fetchone()
        g.user = row


def login_required(view):
    @functools.wraps(view)
    def wrapped(**kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))
        return view(**kwargs)
    return wrapped


def pi_required(view):
    @functools.wraps(view)
    def wrapped(**kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))
        if g.user["role"] != "pi":
            return ("Only the principal investigator can open this page.", 403)
        return view(**kwargs)
    return wrapped


def next_grader_number(db):
    row = db.execute(
        "SELECT COALESCE(MAX(grader_number), 0) + 1 AS n FROM users"
    ).fetchone()
    return row["n"]


def grant_grader_number(db, user_id):
    """Give a user (typically the PI) a grader number so they can author,
    run, and grade cases too. No-op if they already have one."""
    db.execute(
        "UPDATE users SET grader_number = ? WHERE id = ? AND grader_number IS NULL",
        (next_grader_number(db), user_id),
    )
    db.commit()
    return db.execute(
        "SELECT grader_number FROM users WHERE id = ?", (user_id,)
    ).fetchone()["grader_number"]


def create_user(db, name, email, role, invited=True):
    """Insert a user; graders get the next free grader number."""
    grader_number = None
    if role == "grader":
        grader_number = next_grader_number(db)
    token = secrets.token_urlsafe(24) if invited else None
    cursor = db.execute(
        "INSERT INTO users (name, email, role, grader_number, invite_token) "
        "VALUES (?, ?, ?, ?, ?)",
        (name.strip(), email.strip(), role, grader_number, token),
    )
    db.commit()
    return cursor.lastrowid, token


@bp.route("/login", methods=("GET", "POST"))
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        row = get_db().execute(
            "SELECT * FROM users WHERE email = ? AND disabled = 0", (email,)
        ).fetchone()
        if (
            row is None
            or not row["password_hash"]
            or not check_password_hash(row["password_hash"], password)
        ):
            error = "That email and password do not match an account."
        else:
            session.clear()
            session["user_id"] = row["id"]
            return redirect(url_for("index"))
    return render_template("login.html", error=error)


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


@bp.route("/join/<token>", methods=("GET", "POST"))
def join(token):
    db = get_db()
    row = db.execute(
        "SELECT * FROM users WHERE invite_token = ? AND disabled = 0", (token,)
    ).fetchone()
    if row is None:
        return ("This invitation link is not valid (it may already have "
                "been used). Ask the investigator for a new one.", 404)
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        name = request.form.get("name", "").strip() or row["name"]
        if len(password) < 8:
            error = "Please choose a password of at least 8 characters."
        elif password != confirm:
            error = "The two passwords do not match."
        else:
            db.execute(
                "UPDATE users SET name = ?, password_hash = ?, invite_token = NULL "
                "WHERE id = ?",
                (name, generate_password_hash(password), row["id"]),
            )
            db.commit()
            session.clear()
            session["user_id"] = row["id"]
            flash("Welcome! Your account is ready.")
            return redirect(url_for("index"))
    return render_template("join.html", invite=row, error=error)


@bp.route("/people", methods=("GET", "POST"))
@pi_required
def people():
    db = get_db()
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        if not name or "@" not in email:
            error = "A name and a valid email address are needed."
        elif db.execute(
            "SELECT 1 FROM users WHERE email = ?", (email,)
        ).fetchone():
            error = "There is already an account with that email."
        else:
            _user_id, token = create_user(db, name, email, "grader")
            flash(
                "Invite created for {}. Send them this link: {}".format(
                    name, url_for("auth.join", token=token, _external=True)
                )
            )
            return redirect(url_for("auth.people"))
    rows = db.execute(
        "SELECT * FROM users ORDER BY role DESC, name COLLATE NOCASE"
    ).fetchall()
    invite_links = {
        row["id"]: url_for("auth.join", token=row["invite_token"], _external=True)
        for row in rows if row["invite_token"]
    }
    # Friendly accountability: what each person has contributed.
    cases_written = {
        row["owner_id"]: row["n"]
        for row in db.execute(
            "SELECT owner_id, COUNT(*) AS n FROM cases WHERE deleted = 0 "
            "GROUP BY owner_id"
        )
    }
    answers_graded = {
        row["grader_id"]: row["n"]
        for row in db.execute(
            "SELECT ga.grader_id, COUNT(*) AS n FROM grades "
            "JOIN grading_assignments ga ON ga.id = grades.assignment_id "
            "WHERE grades.superseded = 0 AND grades.score IS NOT NULL "
            "GROUP BY ga.grader_id"
        )
    }
    return render_template(
        "people.html", people=rows, invite_links=invite_links, error=error,
        cases_written=cases_written, answers_graded=answers_graded,
    )
