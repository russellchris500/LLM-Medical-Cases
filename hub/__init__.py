"""Study Hub - the web application of the LLM Medical Cases framework.

The hub replaces the emailed-files workflow: graders author and grade
their cases in the browser, run jobs flow to whoever executes them (the
grader's own local Runner, or the PI's), and the PI's ranking dashboard
reads everything live. LLMs themselves are still run by the local Runner
program - browser-site models need a real browser, personal logins, and
a human nearby, which no web server can provide.

Run locally for development:
    python -m hub.manage init-db
    python -m hub.manage create-pi "Your Name" you@example.org
    python -m hub.manage run
"""

import os

from flask import Flask, g, redirect, url_for

from . import db as hub_db


def create_app(instance_dir=None, secret_key=None):
    app = Flask(__name__)
    instance_dir = instance_dir or os.environ.get(
        "STUDYHUB_DATA", os.path.join(os.getcwd(), "hub_data")
    )
    os.makedirs(instance_dir, exist_ok=True)
    app.config["INSTANCE_DIR"] = instance_dir
    app.config["DATABASE"] = os.path.join(instance_dir, "study.db")
    app.config["ANSWER_DIR"] = os.path.join(instance_dir, "answers")
    app.config["SECRET_KEY"] = (
        secret_key
        or os.environ.get("STUDYHUB_SECRET")
        or hub_db.load_or_create_secret(os.path.join(instance_dir, "secret_key"))
    )
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # uploads incl. images
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    hub_db.init_app(app)

    from . import auth, cases, grading, home, rankings, runner_api, runs

    app.register_blueprint(auth.bp)
    app.register_blueprint(cases.bp)
    app.register_blueprint(runs.bp)
    app.register_blueprint(runner_api.bp)
    app.register_blueprint(grading.bp)
    app.register_blueprint(rankings.bp)
    app.register_blueprint(home.bp)

    @app.route("/")
    def index():
        if g.user is None:
            return redirect(url_for("auth.login"))
        return redirect(url_for("home.dashboard"))

    return app
