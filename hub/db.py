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

# Future additive changes: append ("0002", "ALTER TABLE ...") entries.
MIGRATIONS = [
    ("0001", SCHEMA),
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
