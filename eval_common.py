#!/usr/bin/env python3
"""Shared building blocks for Programs 3 and 4 of the LLM Medical Cases
evaluation framework: settings, saved case selections, the answers store,
and the case-selection expression parser.

Requires only the Python 3 standard library.
"""

import copy
import glob
import hashlib
import json
import os
import re
import tempfile

from case_editor import FORMAT_VERSION, CaseStoreError, make_case_id, now_iso

SETTINGS_FILENAME = "settings.json"
CASE_SETS_FILENAME = "case_sets.json"
ANSWERS_FILENAME = "answers.json"
ANSWER_IMAGES_DIR = "answer_images"


def save_json_atomic(path, data, private=False):
    """Write JSON so an interrupted save can't corrupt the file."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        if private:
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_json(path, what):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise CaseStoreError("{} ({}) is not valid JSON: {}".format(path, what, e))


def case_hash(case):
    """Fingerprint of the parts of a case an LLM answer depends on."""
    material = case["case_text"] + "\n" + "\n".join(case["rubric"])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def split_case_id(case_id):
    provider, number = case_id.split("-")
    return int(provider), int(number)


def sort_case_ids(case_ids):
    return sorted(case_ids, key=split_case_id)


# ---------- case-selection expressions ----------
#
# Comma-separated tokens, each one of:
#   003-007            a single case (also accepts 3-7)
#   003-001..003-020   an inclusive range (also 003-001..20, "003-001 to 003-020");
#                      a range must stay within one provider
#   provider 3 / p3    every case from provider 3
#   all                every case
#
# A bare hyphen is never accepted as a range separator because case IDs
# already contain a hyphen.

_SINGLE_RE = re.compile(r"0*(\d+)-0*(\d+)$")
_RANGE_RE = re.compile(r"0*(\d+)-0*(\d+)\s*(?:\.\.+|\bto\b)\s*(?:0*(\d+)-)?0*(\d+)$", re.IGNORECASE)
_PROVIDER_RE = re.compile(r"(?:p|provider)\s*0*(\d+)$", re.IGNORECASE)


class SelectionError(Exception):
    """Raised when a case-selection expression cannot be understood."""


def parse_selection(expression, cases_by_id):
    """Expand a selection expression against the available cases.

    cases_by_id: dict of case_id -> case (e.g. MasterStore.cases).
    Returns (sorted case_ids that exist, warnings) where warnings is a list
    of plain-language notes about requested cases that don't exist.
    Raises SelectionError when a token can't be understood at all.
    """
    available = set(cases_by_id)
    by_provider = {}
    for case_id in available:
        provider, number = split_case_id(case_id)
        by_provider.setdefault(provider, set()).add(number)

    selected = set()
    warnings = []
    for raw_token in expression.split(","):
        token = raw_token.strip()
        if not token:
            continue
        lowered = token.lower()

        if lowered == "all":
            selected |= available
            continue

        match = _PROVIDER_RE.match(token)
        if match:
            provider = int(match.group(1))
            if provider not in by_provider:
                warnings.append("There are no cases from provider {}.".format(provider))
            else:
                selected |= {make_case_id(provider, n) for n in by_provider[provider]}
            continue

        match = _RANGE_RE.match(token)
        if match:
            provider = int(match.group(1))
            start = int(match.group(2))
            end_provider = int(match.group(3)) if match.group(3) else provider
            end = int(match.group(4))
            if end_provider != provider:
                raise SelectionError(
                    "A range must stay within one provider ('{}' goes from provider {} to {}).".format(
                        token, provider, end_provider
                    )
                )
            if end < start:
                start, end = end, start
            in_master = by_provider.get(provider, set())
            hits = [n for n in range(start, end + 1) if n in in_master]
            missing = (end - start + 1) - len(hits)
            if not hits:
                warnings.append(
                    "No cases exist in the range {} (skipped).".format(token)
                )
            elif missing:
                warnings.append(
                    "{} case number{} in the range {} {} not in the database and {} skipped.".format(
                        missing,
                        "" if missing == 1 else "s",
                        token,
                        "is" if missing == 1 else "are",
                        "was" if missing == 1 else "were",
                    )
                )
            selected |= {make_case_id(provider, n) for n in hits}
            continue

        match = _SINGLE_RE.match(token)
        if match:
            case_id = make_case_id(int(match.group(1)), int(match.group(2)))
            if case_id in available:
                selected.add(case_id)
            else:
                warnings.append("Case {} is not in the database (skipped).".format(case_id))
            continue

        raise SelectionError(
            "Could not understand '{}'. Use forms like 003-007, 003-001..003-020, "
            "provider 3, or all.".format(token)
        )

    return sort_case_ids(selected), warnings


# ---------- saved case sets ----------

SET_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,40}$")


class CaseSetStore:
    """Named, frozen case selections (case_sets.json), shared by Programs 3 & 4."""

    def __init__(self, path=CASE_SETS_FILENAME):
        self.path = path
        self.sets = {}  # name -> {expression, case_ids, created_at, updated_at}

    @classmethod
    def load_or_create(cls, path=CASE_SETS_FILENAME):
        store = cls(path)
        if not os.path.exists(path):
            return store
        data = load_json(path, "saved case sets")
        if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
            raise CaseStoreError("{} is not a version-{} case-set file.".format(path, FORMAT_VERSION))
        sets = data.get("case_sets", {})
        if not isinstance(sets, dict):
            raise CaseStoreError("{} has an invalid case_sets section.".format(path))
        for name, entry in sets.items():
            if not (isinstance(entry, dict) and isinstance(entry.get("case_ids"), list)):
                raise CaseStoreError("Case set '{}' in {} is invalid.".format(name, path))
            store.sets[name] = {
                "expression": entry.get("expression", ""),
                "case_ids": [str(c) for c in entry["case_ids"]],
                "created_at": entry.get("created_at", now_iso()),
                "updated_at": entry.get("updated_at", now_iso()),
            }
        return store

    def save(self):
        save_json_atomic(
            self.path, {"format_version": FORMAT_VERSION, "case_sets": self.sets}
        )

    def find(self, name):
        """Case-insensitive lookup returning the stored name, or None."""
        lowered = name.lower()
        for existing in self.sets:
            if existing.lower() == lowered:
                return existing
        return None

    def add(self, name, expression, case_ids):
        if not SET_NAME_RE.match(name):
            raise CaseStoreError(
                "Set names may only use letters, digits, dots, dashes, and "
                "underscores (up to 40 characters)."
            )
        if self.find(name):
            raise CaseStoreError("A case set named '{}' already exists.".format(name))
        if not case_ids:
            raise CaseStoreError("A case set needs at least one case.")
        timestamp = now_iso()
        self.sets[name] = {
            "expression": expression,
            "case_ids": sort_case_ids(case_ids),
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        self.save()

    def rename(self, old_name, new_name):
        stored = self.find(old_name)
        if stored is None:
            raise CaseStoreError("No case set named '{}'.".format(old_name))
        if not SET_NAME_RE.match(new_name):
            raise CaseStoreError(
                "Set names may only use letters, digits, dots, dashes, and "
                "underscores (up to 40 characters)."
            )
        clash = self.find(new_name)
        if clash and clash != stored:
            raise CaseStoreError("A case set named '{}' already exists.".format(new_name))
        entry = self.sets.pop(stored)
        entry["updated_at"] = now_iso()
        self.sets[new_name] = entry
        self.save()

    def delete(self, name):
        stored = self.find(name)
        if stored is None:
            raise CaseStoreError("No case set named '{}'.".format(name))
        del self.sets[stored]
        self.save()

    def resolve(self, name, cases_by_id):
        """Split a saved set into (ids still present, ids no longer present)."""
        stored = self.find(name)
        if stored is None:
            raise CaseStoreError("No case set named '{}'.".format(name))
        wanted = self.sets[stored]["case_ids"]
        present = [c for c in wanted if c in cases_by_id]
        missing = [c for c in wanted if c not in cases_by_id]
        return present, missing


# ---------- settings ----------

DEFAULT_SETTINGS = {
    "format_version": FORMAT_VERSION,
    "api_models": {
        "claude": {"api_key": "", "model": ""},
        "gpt": {"api_key": "", "model": ""},
        "gemini": {"api_key": "", "model": ""},
        "grok": {"api_key": "", "model": ""},
    },
    "browser_models": {
        "openevidence": {"username": "", "password": "", "last_login_ok": None},
        "uptodate": {"username": "", "password": "", "last_login_ok": None},
        "doximity": {"username": "", "password": "", "last_login_ok": None},
    },
    "options": {
        "request_timeout_s": 180,
        "max_retries": 5,
        "browser_question_delay_s": 8,
        "answer_stable_seconds": 10,
        "answer_max_wait_seconds": 300,
        "enable_test_model": False,
    },
}


class SettingsStore:
    """settings.json: API keys, site logins, and tuning options.

    The file is saved with private permissions (0600) because it holds
    credentials. Unknown keys in the file are preserved; missing keys are
    filled from defaults, so upgrading the program never loses settings.
    """

    def __init__(self, path=SETTINGS_FILENAME):
        self.path = path
        self.data = copy.deepcopy(DEFAULT_SETTINGS)

    @classmethod
    def load_or_create(cls, path=SETTINGS_FILENAME):
        store = cls(path)
        if os.path.exists(path):
            data = load_json(path, "settings")
            if not isinstance(data, dict):
                raise CaseStoreError("{} does not contain settings.".format(path))
            store._merge(store.data, data)
        return store

    @staticmethod
    def _merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                SettingsStore._merge(base[key], value)
            else:
                base[key] = value

    def save(self):
        save_json_atomic(self.path, self.data, private=True)

    def api_model(self, model_id):
        return self.data["api_models"].setdefault(model_id, {"api_key": "", "model": ""})

    def browser_model(self, site_id):
        return self.data["browser_models"].setdefault(
            site_id, {"username": "", "password": "", "last_login_ok": None}
        )

    def option(self, name):
        return self.data["options"].get(name, DEFAULT_SETTINGS["options"].get(name))


# ---------- answers ----------

OK_STATUSES = ("ok", "ok_manual")


class AnswersStore:
    """answers.json + answer_images/: one record per (case_id, model_id).

    save() runs after every change, so a crash or Ctrl-C never loses more
    than the answer that was in flight - re-running resumes automatically.
    """

    def __init__(self, path=ANSWERS_FILENAME, images_dir=ANSWER_IMAGES_DIR):
        self.path = path
        self.images_dir = images_dir
        self.prompt_template_version = None
        self.answers = {}  # (case_id, model_id) -> record

    @classmethod
    def load_or_create(cls, path=ANSWERS_FILENAME, images_dir=ANSWER_IMAGES_DIR):
        store = cls(path, images_dir)
        if not os.path.exists(path):
            return store
        data = load_json(path, "answers")
        if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
            raise CaseStoreError("{} is not a version-{} answers file.".format(path, FORMAT_VERSION))
        store.prompt_template_version = data.get("prompt_template_version")
        for record in data.get("answers", []):
            if not (
                isinstance(record, dict)
                and isinstance(record.get("case_id"), str)
                and isinstance(record.get("model_id"), str)
            ):
                raise CaseStoreError("{} contains an invalid answer record.".format(path))
            store.answers[(record["case_id"], record["model_id"])] = record
        return store

    def save(self):
        ordered = sorted(
            self.answers.values(),
            key=lambda r: (split_case_id(r["case_id"]), r["model_id"]),
        )
        save_json_atomic(
            self.path,
            {
                "format_version": FORMAT_VERSION,
                "prompt_template_version": self.prompt_template_version,
                "answers": ordered,
            },
        )

    def get(self, case_id, model_id):
        return self.answers.get((case_id, model_id))

    def upsert(self, record):
        self.answers[(record["case_id"], record["model_id"])] = record
        self.save()

    def model_ids(self):
        return sorted({model_id for _, model_id in self.answers})

    def answered_case_ids(self, ok_only=True):
        ids = {
            case_id
            for (case_id, _), record in self.answers.items()
            if not ok_only or record.get("status") in OK_STATUSES
        }
        return sort_case_ids(ids)

    def image_basename(self, case_id, model_id):
        return "{}_{}".format(case_id, model_id)

    def clear_images(self, case_id, model_id):
        """Delete this pair's image files before a re-run writes new ones."""
        pattern = os.path.join(
            self.images_dir, self.image_basename(case_id, model_id) + "_*"
        )
        for path in glob.glob(pattern):
            try:
                os.unlink(path)
            except OSError:
                pass

    def ensure_images_dir(self):
        os.makedirs(self.images_dir, exist_ok=True)
        return self.images_dir
