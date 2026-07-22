"""SQLite layer for the Study Hub: schema, connections, tiny helpers.

One file (study.db) holds the whole study except answer images, which
live on disk under the instance directory. The schema is created with
CREATE TABLE IF NOT EXISTS so init-db is safe to re-run; additive schema
changes ship as numbered migration steps in MIGRATIONS.
"""

import os
import secrets
import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('pi', 'grader')),
    grader_number INTEGER UNIQUE,          -- NNN in case IDs GNNN-CCC
    password_hash TEXT,                    -- NULL until the invite is used
    invite_token TEXT UNIQUE,              -- NULL once the account is set up
    max_assigned_case_number INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    disabled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runner_tokens (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    token_hash TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    last_seen TEXT,
    revoked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,                   -- e.g. G007-012
    owner_id INTEGER NOT NULL REFERENCES users(id),
    case_number INTEGER NOT NULL,
    case_text TEXT NOT NULL,
    rubric TEXT NOT NULL,                  -- JSON list of items
    rubric_version INTEGER NOT NULL DEFAULT 1,
    rubric_history TEXT NOT NULL DEFAULT '[]',  -- JSON list of edits
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    deleted INTEGER NOT NULL DEFAULT 0,
    UNIQUE (owner_id, case_number)
);
"""

SCHEMA_0002 = """
CREATE TABLE IF NOT EXISTS run_jobs (
    id INTEGER PRIMARY KEY,
    requested_by INTEGER NOT NULL REFERENCES users(id),
    assigned_to INTEGER NOT NULL REFERENCES users(id),
    case_ids TEXT NOT NULL,                -- JSON list of case IDs
    llm_ids TEXT NOT NULL,                 -- JSON list; the runner resolves
                                           -- model names from ITS settings
    note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'done', 'failed', 'cancelled')),
    status_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(id),
    run_job_id INTEGER REFERENCES run_jobs(id),
    run_by INTEGER NOT NULL REFERENCES users(id),
    llm_id TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    variant_id TEXT NOT NULL,              -- llm@model-name scored identity
    model_display_name TEXT NOT NULL DEFAULT '',
    response_text TEXT NOT NULL DEFAULT '',
    answer_html_path TEXT,
    image_paths TEXT NOT NULL DEFAULT '[]',
    thinking_setting TEXT NOT NULL DEFAULT '',
    model_reported TEXT NOT NULL DEFAULT '',
    deep_thinking INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'ok',
    error TEXT,
    case_text_sha256 TEXT NOT NULL DEFAULT '',
    rubric_version_at_run INTEGER NOT NULL DEFAULT 1,
    run_by_owner INTEGER NOT NULL DEFAULT 0,  -- blinding honesty flag
    self_id_warning TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (case_id, variant_id, run_by)
);
"""

SCHEMA_0003 = """
CREATE TABLE IF NOT EXISTS grading_assignments (
    id INTEGER PRIMARY KEY,
    grader_id INTEGER NOT NULL REFERENCES users(id),
    case_id TEXT NOT NULL REFERENCES cases(id),
    kind TEXT NOT NULL DEFAULT 'own' CHECK (kind IN ('own', 'cross')),
    assigned_by INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (grader_id, case_id)
);

-- The per-grader blinding: which shuffled letter maps to which answer.
-- Created server-side, never sent to anyone; stable once assigned.
CREATE TABLE IF NOT EXISTS blind_labels (
    id INTEGER PRIMARY KEY,
    assignment_id INTEGER NOT NULL REFERENCES grading_assignments(id),
    label TEXT NOT NULL,
    answer_id INTEGER NOT NULL REFERENCES answers(id),
    UNIQUE (assignment_id, label),
    UNIQUE (assignment_id, answer_id)
);

CREATE TABLE IF NOT EXISTS grades (
    id INTEGER PRIMARY KEY,
    assignment_id INTEGER NOT NULL REFERENCES grading_assignments(id),
    answer_id INTEGER NOT NULL REFERENCES answers(id),
    rubric_results TEXT NOT NULL,          -- JSON list of booleans
    unnecessary_risk INTEGER,              -- NULL = never applied
    poor_approach INTEGER,                 -- NULL = never applied
    score INTEGER NOT NULL CHECK (score IN (0, 1, 2)),
    comment TEXT NOT NULL DEFAULT '',
    rubric_version INTEGER NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    superseded_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS grades_one_active
    ON grades (assignment_id, answer_id) WHERE superseded = 0;

CREATE TABLE IF NOT EXISTS rubric_flags (
    id INTEGER PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(id),
    item_index INTEGER NOT NULL,
    item_text TEXT NOT NULL,
    note TEXT NOT NULL,
    flagged_by INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

SCHEMA_0004 = """
CREATE TABLE IF NOT EXISTS ranking_snapshots (
    id INTEGER PRIMARY KEY,
    created_by INTEGER NOT NULL REFERENCES users(id),
    summary TEXT NOT NULL DEFAULT '',
    results TEXT NOT NULL,                 -- JSON: ratings, stats, agreement
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

SCHEMA_0005 = """
ALTER TABLE answers ADD COLUMN discarded_by INTEGER REFERENCES users(id);
ALTER TABLE answers ADD COLUMN discarded_reason TEXT NOT NULL DEFAULT '';
"""

# Future additive changes: append ("0006", "ALTER TABLE ...") entries.
MIGRATIONS = [
    ("0001", SCHEMA),
    ("0002", SCHEMA_0002),
    ("0003", SCHEMA_0003),
    ("0004", SCHEMA_0004),
    ("0005", SCHEMA_0005),
]


def load_or_create_secret(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            secret = f.read().strip()
        if secret:
            return secret
    except OSError:
        pass
    secret = secrets.token_hex(32)
    with open(path, "w", encoding="utf-8") as f:
        f.write(secret)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return secret


def connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_db():
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE"])
    return g.db


def close_db(_exception=None):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db(path):
    connection = connect(path)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL "
            "DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')))"
        )
        done = {
            row["version"]
            for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for version, statements in MIGRATIONS:
            if version in done:
                continue
            connection.executescript(statements)
            connection.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)", (version,)
            )
        connection.commit()
    finally:
        connection.close()


def init_app(app):
    app.teardown_appcontext(close_db)
    init_db(app.config["DATABASE"])
