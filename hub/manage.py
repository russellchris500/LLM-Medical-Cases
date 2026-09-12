"""Study Hub management commands (run from the repository folder):

    python -m hub.manage init-db
    python -m hub.manage create-pi "Dr Name" pi@example.org
    python -m hub.manage import-legacy grader@example.org [folder]
    python -m hub.manage make-grader email@example.org
    python -m hub.manage judge-run RUN_ID
    python -m hub.manage run           (development server on port 5000)

make-grader gives an account (typically the PI's) a grader number so it
can author, run, and grade cases too - the same thing that happens
automatically the first time the PI clicks New case.

judge-run executes (or resumes) an AI-judge run in the foreground,
printing progress - for runs the web process could not finish (a
restart mid-run) or for very large runs. Finished verdicts are kept.

import-legacy moves an existing desktop-programs study into the hub:
cases from master_cases.json, collected answers (with their image
files), and grades from scores_*.json + the key files in
scoring_packages/. Everything is owned by the given grader account.

STUDYHUB_DATA sets where study.db and answer files live (default:
./hub_data next to where you run the command)."""

import getpass
import glob
import json
import os
import shutil
import sys

from werkzeug.security import generate_password_hash

from . import create_app
from .auth import create_user
from .db import connect


def import_legacy(app, grader_email, folder="."):
    """Returns a report dict; raises SystemExit with a message on setup
    problems (unknown grader, missing master file)."""
    from .auth import grant_grader_number

    db = connect(app.config["DATABASE"])
    try:
        grader = db.execute(
            "SELECT * FROM users WHERE email = ?", (grader_email,)
        ).fetchone()
        if grader is None:
            raise SystemExit(
                "No account with email {} - create it first (create-pi, or an "
                "invite on the People page).".format(grader_email)
            )
        if grader["grader_number"] is None:
            # A PI who has not opted into grading yet: the import makes
            # them a grader, same as their first New case click would.
            number = grant_grader_number(db, grader["id"])
            print("{} is now also grader {} (the imported cases need a "
                  "grader owner).".format(grader["name"], number))
            grader = db.execute(
                "SELECT * FROM users WHERE id = ?", (grader["id"],)
            ).fetchone()
        master_path = os.path.join(folder, "master_cases.json")
        if not os.path.exists(master_path):
            raise SystemExit("No master_cases.json in {}.".format(folder))
        report = {"cases": 0, "answers": 0, "grades": 0, "skipped": []}

        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)
        for case in master.get("cases", []):
            db.execute(
                "INSERT OR IGNORE INTO cases (id, owner_id, case_number, "
                "case_text, rubric, rubric_version, rubric_history, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (case["case_id"], grader["id"], case["case_number"],
                 case["case_text"], json.dumps(case["rubric"]),
                 case.get("rubric_version", 1),
                 json.dumps(case.get("rubric_history", [])),
                 case.get("created_at", ""), case.get("updated_at", "")),
            )
            report["cases"] += 1

        answers_path = os.path.join(folder, "answers.json")
        if os.path.exists(answers_path):
            with open(answers_path, "r", encoding="utf-8") as f:
                answers = json.load(f)
            for record in answers.get("answers", []):
                if record.get("status") not in ("ok", "ok_manual"):
                    continue
                image_paths = []
                subdir = "{}_{}".format(record["case_id"], grader["id"])
                for source in record.get("images", []):
                    source_path = os.path.join(folder, source)
                    if not os.path.exists(source_path):
                        continue
                    target_dir = os.path.join(app.config["ANSWER_DIR"], subdir)
                    os.makedirs(target_dir, exist_ok=True)
                    target = os.path.join(target_dir, os.path.basename(source))
                    shutil.copyfile(source_path, target)
                    image_paths.append(
                        os.path.relpath(target, app.config["ANSWER_DIR"])
                    )
                db.execute(
                    "INSERT INTO answers (case_id, run_by, llm_id, model_name, "
                    "variant_id, model_display_name, response_text, image_paths, "
                    "thinking_setting, model_reported, deep_thinking, status, "
                    "case_text_sha256, rubric_version_at_run, run_by_owner) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT (case_id, variant_id, run_by) DO NOTHING",
                    (record["case_id"], grader["id"],
                     record.get("llm_id", record["model_id"]),
                     record.get("model_name", ""), record["model_id"],
                     record.get("model_display_name", record["model_id"]),
                     record.get("response_text", ""), json.dumps(image_paths),
                     record.get("thinking_setting", ""),
                     record.get("model_reported", ""),
                     1 if record.get("deep_thinking", True) else 0,
                     record.get("status", "ok"),
                     record.get("case_text_sha256", ""),
                     record.get("rubric_version", 1)),
                )
                report["answers"] += 1

        # Grades: scores files joined through the package key files. The
        # legacy scorer saves scores_<package>.json NEXT TO THE ZIP it
        # graded - often in scoring_packages/ or a copy of it - so search
        # the study folder root AND every first-level subfolder for both.
        search_dirs = [folder] + sorted(
            entry.path for entry in os.scandir(folder) if entry.is_dir()
        )
        keys = {}
        for directory in search_dirs:
            for key_path in glob.glob(
                os.path.join(directory, "*_KEY_DO_NOT_SEND.json")
            ):
                with open(key_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("package_id"):
                    keys[data["package_id"]] = data.get("key", {})
        scores_paths = []
        seen_paths = set()
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, "scores_*.json"))):
                real = os.path.realpath(path)
                if real not in seen_paths:
                    seen_paths.add(real)
                    scores_paths.append(path)
        if not scores_paths:
            report["skipped"].append(
                "No scores_*.json files were found in {} or its subfolders - "
                "if you graded answers, copy the scores file(s) the scorer "
                "program saved (next to the zip it graded) into this folder "
                "and re-run the import; re-running is safe.".format(
                    os.path.abspath(folder)
                )
            )
        for scores_path in scores_paths:
            with open(scores_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = keys.get(data.get("package_id"))
            if key is None:
                report["skipped"].append(
                    "{}: no matching *_KEY_DO_NOT_SEND.json was found for its "
                    "package - its grades were left out".format(
                        os.path.basename(scores_path)
                    )
                )
                continue
            imported_here = 0
            missing_answers = 0
            for record in data.get("scores", []):
                entry = (key.get(record["case_id"]) or {}).get(record["label"])
                if entry is None:
                    continue
                answer = db.execute(
                    "SELECT id FROM answers WHERE case_id = ? AND variant_id = ? "
                    "AND run_by = ?",
                    (record["case_id"], entry["model_id"], grader["id"]),
                ).fetchone()
                if answer is None:
                    missing_answers += 1
                    continue
                db.execute(
                    "INSERT OR IGNORE INTO grading_assignments "
                    "(grader_id, case_id, kind) VALUES (?, ?, 'own')",
                    (grader["id"], record["case_id"]),
                )
                assignment = db.execute(
                    "SELECT id FROM grading_assignments WHERE grader_id = ? "
                    "AND case_id = ?",
                    (grader["id"], record["case_id"]),
                ).fetchone()
                existing = db.execute(
                    "SELECT id FROM grades WHERE assignment_id = ? AND "
                    "answer_id = ? AND superseded = 0",
                    (assignment["id"], answer["id"]),
                ).fetchone()
                if existing is not None:
                    continue
                db.execute(
                    "INSERT INTO grades (assignment_id, answer_id, "
                    "rubric_results, unnecessary_risk, poor_approach, score, "
                    "comment, rubric_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (assignment["id"], answer["id"],
                     json.dumps(record.get("rubric_results", [])),
                     record.get("unnecessary_risk"),
                     record.get("poor_approach"), record.get("score", 0),
                     record.get("comment", ""), record.get("rubric_version", 1)),
                )
                report["grades"] += 1
                imported_here += 1
            note = "{}: {} grade(s) imported".format(
                os.path.basename(scores_path), imported_here
            )
            if missing_answers:
                note += (", {} skipped (no matching imported answer - was "
                         "answers.json imported from the same study?)"
                         .format(missing_answers))
            report.setdefault("details", []).append(note)
        db.commit()
        return report
    finally:
        db.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    command = argv[0]
    app = create_app()

    if command == "init-db":
        # create_app already ran the migrations.
        print("Database ready at {}".format(app.config["DATABASE"]))
        return 0

    if command == "create-pi":
        if len(argv) != 3:
            print('Usage: python -m hub.manage create-pi "Name" email@example.org')
            return 1
        name, email = argv[1], argv[2]
        password = getpass.getpass("Choose the PI password (8+ characters): ")
        if len(password) < 8:
            print("Password too short.")
            return 1
        db = connect(app.config["DATABASE"])
        try:
            if db.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                print("There is already an account with that email.")
                return 1
            user_id, _token = create_user(db, name, email, "pi", invited=False)
            db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (generate_password_hash(password), user_id),
            )
            db.commit()
        finally:
            db.close()
        print("PI account created for {} <{}>.".format(name, email))
        return 0

    if command == "make-grader":
        if len(argv) != 2:
            print("Usage: python -m hub.manage make-grader email@example.org")
            return 1
        from .auth import grant_grader_number

        db = connect(app.config["DATABASE"])
        try:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?", (argv[1],)
            ).fetchone()
            if user is None:
                print("No account with that email.")
                return 1
            number = grant_grader_number(db, user["id"])
        finally:
            db.close()
        print("{} is grader {} (case IDs G{:03d}-...).".format(
            user["name"], number, number
        ))
        return 0

    if command == "import-legacy":
        if len(argv) not in (2, 3):
            print("Usage: python -m hub.manage import-legacy grader@example.org "
                  "[folder]")
            return 1
        folder = argv[2] if len(argv) == 3 else "."
        report = import_legacy(app, argv[1], folder)
        print("Imported {} case(s), {} answer(s), {} grade(s).".format(
            report["cases"], report["answers"], report["grades"]
        ))
        for line in report.get("details", []):
            print("  " + line)
        for line in report["skipped"]:
            print("Note: " + line)
        return 0

    if command == "judge-run":
        if len(argv) != 2 or not argv[1].isdigit():
            print("Usage: python -m hub.manage judge-run RUN_ID")
            return 1
        from .judge import execute_run

        db = connect(app.config["DATABASE"])
        try:
            run = db.execute(
                "SELECT * FROM judge_runs WHERE id = ?", (int(argv[1]),)
            ).fetchone()
            if run is None:
                print("No judge run with that id.")
                return 1
            if run["status"] not in ("queued", "running", "stopped",
                                     "failed"):
                print("Run {} is {} - nothing to do.".format(
                    run["id"], run["status"]))
                return 0
            # A stopped/failed run resumes: only_missing skips what is done.
            db.execute(
                "UPDATE judge_runs SET status = 'queued', note = '' "
                "WHERE id = ?", (run["id"],),
            )
            db.commit()
        finally:
            db.close()
        execute_run(app.config["DATABASE"], int(argv[1]), log=print)
        db = connect(app.config["DATABASE"])
        try:
            run = db.execute(
                "SELECT * FROM judge_runs WHERE id = ?", (int(argv[1]),)
            ).fetchone()
        finally:
            db.close()
        print("Run {}: {}. {}".format(run["id"], run["status"], run["note"]))
        return 0 if run["status"] == "done" else 1

    if command == "run":
        app.run(host="127.0.0.1", port=5000, debug=False)
        return 0

    print("Unknown command: {}".format(command))
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
