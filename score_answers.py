#!/usr/bin/env python3
"""Program 5 of the LLM Medical Cases evaluation framework: the Answer Scorer.

A scorer receives a scoring package (a .zip built by the PI with Program 4)
by email and grades the anonymized AI answers in it against each case's
rubric. Each answer gets a score of 0, 1, or 2:

  0 - any rubric item is missed, OR the answer takes unnecessary risk
      with the patient (even if every item is covered)
  1 - every rubric item is covered, but the approach is poor
  2 - every rubric item is covered and the approach is acceptable

It is a normal window-based program: double-click it (or run:
py score_answers.py) and it first asks WHERE the zip you were emailed is
saved - point it at that folder (for example your Downloads folder).

Your grades are saved to scores_<package name>.json in the same folder as
the zip, after every answer, so you can stop anytime and continue later.
When you have graded everything, email that scores file back to the
principal investigator.

This file is deliberately self-contained (Python 3.8+ standard library
only, no other files from the project needed), so the PI can email a
scorer just the zip and this one program.
"""

import html as html_escape
import json
import os
import re
import tempfile
import webbrowser
import zipfile
from datetime import datetime, timezone

FORMAT_VERSION = 1

HTML_PAGE_STYLE = (
    "body{font-family:'Segoe UI',Arial,sans-serif;max-width:850px;"
    "margin:2em auto;padding:0 1em;line-height:1.5;color:#222}"
    "pre{background:#f4f4f4;padding:10px;overflow-x:auto}"
    "code{background:#f4f4f4;padding:1px 4px}"
    "h2,h3,h4{margin:1.2em 0 0.4em}"
)


def markdown_to_html(text, title="Answer"):
    """A small renderer for the markdown-style text the models produce
    (headings, **bold**, `code`, bullet and numbered lists, fenced code
    blocks, simple | tables), so an answer can be read formatted in a web
    browser instead of as a wall of symbols. (Copied from eval_common.py;
    this program must stay self-contained.)"""

    def inline(s):
        s = html_escape.escape(s, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return s

    out = []
    mode = [None]  # None | "ul" | "ol" | "pre" | "table"

    def close():
        if mode[0] in ("ul", "ol"):
            out.append("</{}>".format(mode[0]))
        elif mode[0] in ("pre", "table"):
            out.append("</pre>")
        mode[0] = None

    for line in (text or "").splitlines():
        stripped = line.strip()
        if mode[0] == "pre" and not stripped.startswith("```"):
            out.append(html_escape.escape(line))
            continue
        if stripped.startswith("```"):
            if mode[0] == "pre":
                close()
            else:
                close()
                out.append("<pre>")
                mode[0] = "pre"
            continue
        heading = re.match(r"(#{1,4})\s+(.+)", stripped)
        if heading:
            close()
            level = min(len(heading.group(1)) + 1, 4)
            out.append("<h{0}>{1}</h{0}>".format(level, inline(heading.group(2))))
            continue
        if re.match(r"[-*•]\s+", stripped):
            if mode[0] != "ul":
                close()
                out.append("<ul>")
                mode[0] = "ul"
            out.append("<li>{}</li>".format(inline(re.sub(r"^[-*•]\s+", "", stripped))))
            continue
        if re.match(r"\d+[.)]\s+", stripped):
            if mode[0] != "ol":
                close()
                out.append("<ol>")
                mode[0] = "ol"
            out.append("<li>{}</li>".format(inline(re.sub(r"^\d+[.)]\s+", "", stripped))))
            continue
        if stripped.startswith("|"):
            if mode[0] != "table":
                close()
                out.append("<pre>")
                mode[0] = "table"
            out.append(html_escape.escape(line))
            continue
        if not stripped:
            close()
            continue
        close()
        out.append("<p>{}</p>".format(inline(stripped)))
    close()
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>{}</title>"
        "<style>{}</style></head><body>{}</body></html>"
    ).format(html_escape.escape(title), HTML_PAGE_STYLE, "\n".join(out))


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def prompt(message):
    try:
        return input(message)
    except EOFError:
        print()
        raise SystemExit(0)


def save_json_atomic(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


class PackageError(Exception):
    """The zip is not a readable scoring package."""


class Package:
    """A scoring package zip: blinded cases, answers, and images."""

    def __init__(self, zip_path, manifest):
        self.zip_path = zip_path
        self.manifest = manifest
        self.cases = manifest.get("cases", [])

    @classmethod
    def load(cls, zip_path):
        try:
            with zipfile.ZipFile(zip_path) as bundle:
                manifest = json.loads(bundle.read("package.json").decode("utf-8"))
        except (OSError, zipfile.BadZipFile, KeyError, ValueError) as e:
            raise PackageError("{} could not be read as a scoring package ({}).".format(
                zip_path, e
            ))
        if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
            raise PackageError("{} is not a version-{} scoring package.".format(
                zip_path, FORMAT_VERSION
            ))
        for case in manifest.get("cases", []):
            if not (
                isinstance(case, dict)
                and isinstance(case.get("case_id"), str)
                and isinstance(case.get("case_text"), str)
                and isinstance(case.get("rubric"), list)
                and isinstance(case.get("answers"), list)
            ):
                raise PackageError("{} contains an invalid case entry.".format(zip_path))
        return cls(zip_path, manifest)

    def all_answers(self):
        """Yield (case, answer) pairs in package order."""
        for case in self.cases:
            for answer in case["answers"]:
                yield case, answer

    def find(self, case_id, label):
        for case in self.cases:
            if case["case_id"] == case_id:
                for answer in case["answers"]:
                    if answer["label"].upper() == label.upper():
                        return case, answer
        return None, None

    def part_note(self):
        part = self.manifest.get("part")
        of = self.manifest.get("of")
        if part and of:
            return " (part {} of {} - each part is scored separately)".format(part, of)
        return ""

    def extract_images(self, answer, out_dir):
        """Copy an answer's images out of the zip; returns their paths."""
        paths = []
        if not answer.get("images"):
            return paths
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path) as bundle:
            for name in answer["images"]:
                target = os.path.join(out_dir, os.path.basename(name))
                try:
                    with bundle.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    paths.append(target)
                except (KeyError, OSError):
                    print("  (image {} is missing from the package)".format(name))
        return paths


class ScoresStore:
    """scores_<package>.json: one grade per (case_id, label), saved after
    every answer so stopping and resuming is always safe."""

    def __init__(self, path, package_manifest):
        self.path = path
        self.package_id = package_manifest.get("package_id", "")
        self.package_name = package_manifest.get("package_name", "")
        self.scorer = ""
        self.created_at = now_iso()
        self.scores = {}  # (case_id, label) -> record
        # Rubric items the scorer flagged as too hard/wrong, for the PI.
        self.rubric_flags = []
        # Grades invalidated by a PI rubric change - kept for the audit
        # trail, no longer counted anywhere.
        self.superseded = []

    @classmethod
    def load_or_create(cls, path, package_manifest):
        store = cls(path, package_manifest)
        if not os.path.exists(path):
            return store
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise PackageError("{} is not valid JSON: {}".format(path, e))
        if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
            raise PackageError("{} is not a version-{} scores file.".format(path, FORMAT_VERSION))
        if data.get("package_id") and data["package_id"] != store.package_id:
            raise PackageError(
                "{} belongs to a different package. Move or rename it, or put "
                "this program in a folder of its own with the right zip.".format(path)
            )
        store.scorer = data.get("scorer", "")
        store.created_at = data.get("created_at", store.created_at)
        store.rubric_flags = [
            f for f in data.get("rubric_flags", []) if isinstance(f, dict)
        ]
        store.superseded = [
            r for r in data.get("superseded", []) if isinstance(r, dict)
        ]
        for record in data.get("scores", []):
            if not (
                isinstance(record, dict)
                and isinstance(record.get("case_id"), str)
                and isinstance(record.get("label"), str)
                and isinstance(record.get("rubric_results"), list)
            ):
                raise PackageError("{} contains an invalid score record.".format(path))
            if "score" not in record:
                # Grades saved by an older version: fill in the 0/1/2 score.
                record.setdefault("unnecessary_risk", None)
                record.setdefault("poor_approach", None)
                record["score"] = compute_score(
                    record["rubric_results"],
                    record["unnecessary_risk"],
                    record["poor_approach"],
                )
            store.scores[(record["case_id"], record["label"])] = record
        return store

    def save(self):
        ordered = sorted(self.scores.values(), key=lambda r: (r["case_id"], r["label"]))
        save_json_atomic(
            self.path,
            {
                "format_version": FORMAT_VERSION,
                "package_id": self.package_id,
                "package_name": self.package_name,
                "scorer": self.scorer,
                "created_at": self.created_at,
                "updated_at": now_iso(),
                "scores": ordered,
                "rubric_flags": self.rubric_flags,
                "superseded": self.superseded,
            },
        )

    def get(self, case_id, label):
        return self.scores.get((case_id, label))

    def upsert(self, case_id, label, rubric_results, unnecessary_risk, poor_approach,
               comment="", rubric_version=1):
        self.scores[(case_id, label)] = {
            "case_id": case_id,
            "label": label,
            "rubric_results": list(rubric_results),
            "unnecessary_risk": unnecessary_risk,
            "poor_approach": poor_approach,
            "score": compute_score(rubric_results, unnecessary_risk, poor_approach),
            "comment": comment,
            # Which rubric this grade was made against.
            "rubric_version": rubric_version,
            "scored_at": now_iso(),
        }
        self.save()

    def add_flag(self, case_id, item_index, item_text, note):
        self.rubric_flags.append({
            "case_id": case_id,
            "item_index": item_index,
            "item_text": item_text,
            "note": note,
            "flagged_at": now_iso(),
        })
        self.save()


def compute_score(rubric_results, unnecessary_risk, poor_approach):
    """The 0/1/2 scoring rule.

    0 - a rubric item is missed, or unnecessary risk was taken
    1 - everything covered but the approach is poor
    2 - everything covered, approach acceptable
    unnecessary_risk / poor_approach are None when the question never
    applied (a missed item already forced the score to 0).
    """
    if not all(rubric_results):
        return 0
    if unnecessary_risk:
        return 0
    if poor_approach:
        return 1
    return 2


# ---------- rubric updates from the PI ----------
#
# When the PI fixes a rubric mid-study, they email a small
# rubric_update_*.json file. Saved next to the zip, it is applied at every
# start: the package's rubric is replaced, and exactly the grades the
# change invalidates are set aside (kept under "superseded" in the scores
# file) and re-queued for grading. Grades are never silently altered
# except the lossless case: an item REMOVED that the answer had covered.


def find_rubric_updates(folder="."):
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(
        os.path.join(folder, n) for n in names
        if re.fullmatch(r"rubric_update_.*\.json", n)
    )


def load_rubric_updates(paths):
    """Merge update files into {case_id: entry}, newest version winning."""
    updates = {}
    problems = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as error:
            problems.append("{}: {}".format(os.path.basename(path), error))
            continue
        if not isinstance(data, dict) or data.get("kind") != "rubric_update":
            problems.append("{}: not a rubric update file".format(os.path.basename(path)))
            continue
        for entry in data.get("cases", []):
            if not (
                isinstance(entry, dict)
                and isinstance(entry.get("case_id"), str)
                and isinstance(entry.get("rubric"), list)
                and isinstance(entry.get("rubric_version"), int)
            ):
                continue
            current = updates.get(entry["case_id"])
            if current is None or entry["rubric_version"] > current["rubric_version"]:
                updates[entry["case_id"]] = entry
    return updates, problems


def grade_survives_removal(record, removed_indexes):
    """A removed rubric item invalidates a grade only when the answer had
    covered everything EXCEPT removed item(s): its score would now depend
    on the risk/approach questions that were never asked. Every other
    grade can be carried over losslessly."""
    results = record.get("rubric_results", [])
    missed_removed = any(
        index < len(results) and results[index] is False for index in removed_indexes
    )
    covered_rest = all(
        result for index, result in enumerate(results) if index not in removed_indexes
    )
    return not (missed_removed and covered_rest)


def carry_over_removal(record, removed_indexes, new_version):
    """Rewrite a surviving grade for the shorter rubric."""
    results = record.get("rubric_results", [])
    new_results = [
        result for index, result in enumerate(results) if index not in removed_indexes
    ]
    record["rubric_results"] = new_results
    record["score"] = compute_score(
        new_results, record.get("unnecessary_risk"), record.get("poor_approach")
    )
    record["rubric_version"] = new_version
    return record


def apply_rubric_updates(package, scores, updates):
    """Apply PI rubric updates to the loaded package and grades.

    Mutates the package's cases (new rubric + version) and the scores
    store (invalidated grades move to superseded). Returns a list of
    plain-language messages for the scorer. Idempotent: applying the same
    updates twice changes nothing the second time.
    """
    messages = []
    changed = False
    for case in package.cases:
        entry = updates.get(case["case_id"])
        if entry is None:
            continue
        old_version = case.get("rubric_version", 1)
        new_version = entry["rubric_version"]
        if new_version > old_version:
            case["rubric"] = [str(item) for item in entry["rubric"]]
            case["rubric_version"] = new_version
            messages.append(
                "Case {}: the investigator updated the rubric (now version {})."
                .format(case["case_id"], new_version)
            )
        # Reconcile this case's grades regardless (grades may predate the
        # update even when the package copy is already current).
        ops = entry.get("ops")
        from_version = entry.get("from_version")
        removed = list((ops or {}).get("removed", []))
        clean_removal_only = (
            ops is not None
            and not (ops.get("added") or ops.get("reworded"))
        )
        requeued = 0
        carried = 0
        for key in list(scores.scores):
            record_case_id, _label = key
            if record_case_id != case["case_id"]:
                continue
            record = scores.scores[key]
            grade_version = record.get("rubric_version", 1)
            if grade_version >= new_version:
                continue
            if (
                clean_removal_only
                and grade_version == from_version
                and grade_survives_removal(record, removed)
            ):
                carry_over_removal(record, removed, new_version)
                carried += 1
                changed = True
                continue
            superseded = dict(record)
            superseded["superseded"] = True
            superseded["superseded_reason"] = (
                "rubric changed to version {}".format(new_version)
            )
            scores.superseded.append(superseded)
            del scores.scores[key]
            requeued += 1
            changed = True
        if requeued or carried:
            parts = []
            if requeued:
                parts.append(
                    "{} answer{} need{} to be graded again".format(
                        requeued, "" if requeued == 1 else "s",
                        "s" if requeued == 1 else "",
                    )
                )
            if carried:
                parts.append(
                    "{} grade{} carried over unchanged".format(
                        carried, "" if carried == 1 else "s"
                    )
                )
            messages.append("Case {}: {}.".format(case["case_id"], " and ".join(parts)))
    if changed:
        scores.save()
    return messages


# ---------- finding the files ----------


def find_package_zips(folder="."):
    """Zips in the folder that look like scoring packages."""
    found = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".zip"):
            continue
        path = os.path.join(folder, name)
        try:
            with zipfile.ZipFile(path) as bundle:
                if "package.json" in bundle.namelist():
                    found.append(name)
        except (OSError, zipfile.BadZipFile):
            continue
    return found


def scores_path_for(zip_name):
    stem = re.sub(r"\.zip$", "", zip_name, flags=re.IGNORECASE)
    return "scores_{}.json".format(stem)


def images_dir_for(zip_name):
    stem = re.sub(r"\.zip$", "", zip_name, flags=re.IGNORECASE)
    return "{}_images".format(stem)


# ---------- grading model (no window code; unit-testable) ----------


class AnswerGrader:
    """Holds one answer's in-progress judgments; the window binds to this.

    The score is 0/1/2: 0 when any rubric item is missed or unnecessary
    risk was taken, 1 when everything is covered but the approach is poor,
    2 otherwise. The risk question only applies once every item is
    covered; the approach question only applies once risk is answered No.
    """

    def __init__(self, rubric, previous=None):
        previous = previous or {}
        stored = previous.get("rubric_results", [])
        self.rubric = list(rubric)
        self.results = [
            stored[i] if i < len(stored) else None for i in range(len(rubric))
        ]
        self.unnecessary_risk = previous.get("unnecessary_risk")
        self.poor_approach = previous.get("poor_approach")
        self.comment = previous.get("comment", "")

    def set_item(self, index, met):
        self.results[index] = bool(met)

    def all_items_answered(self):
        return None not in self.results

    def all_items_met(self):
        return self.all_items_answered() and all(self.results)

    def risk_applies(self):
        return self.all_items_met()

    def poor_applies(self):
        return self.risk_applies() and self.unnecessary_risk is False

    def complete(self):
        if not self.all_items_answered():
            return False
        if not self.all_items_met():
            return True
        if self.unnecessary_risk is None:
            return False
        if self.unnecessary_risk:
            return True
        return self.poor_approach is not None

    def normalized(self):
        """(rubric_results, unnecessary_risk, poor_approach) with the
        questions that never applied set to None."""
        if not self.all_items_met():
            return list(self.results), None, None
        if self.unnecessary_risk:
            return list(self.results), True, None
        return list(self.results), False, self.poor_approach

    def score(self):
        results, risk, poor = self.normalized()
        return compute_score(results, risk, poor)

    def explanation(self):
        if not self.complete():
            missing = []
            if not self.all_items_answered():
                missing.append("answer Covered/Missed for every rubric item")
            elif self.risk_applies() and self.unnecessary_risk is None:
                missing.append("answer the unnecessary-risk question")
            elif self.poor_applies() and self.poor_approach is None:
                missing.append("answer the poor-approach question")
            return "To finish: " + " and ".join(missing) + "."
        score = self.score()
        if score == 2:
            return "Score 2 - all items covered, sound approach."
        if score == 1:
            return "Score 1 - all items covered but the approach is poor."
        if not self.all_items_met():
            missed = sum(1 for r in self.results if r is False)
            return "Score 0 - {} rubric item{} missed.".format(
                missed, "" if missed == 1 else "s"
            )
        return "Score 0 - unnecessary risk to the patient."


# ---------- window interface ----------
#
# Tkinter ships with Python on Windows, so this stays a one-file program a
# scorer can just double-click next to the emailed zip.

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, scrolledtext, ttk

    TK_AVAILABLE = True
except ImportError:  # pragma: no cover - servers without Tk still run tests
    TK_AVAILABLE = False


if TK_AVAILABLE:

    class ScorerApp:
        def __init__(self, root, package, scores):
            self.root = root
            self.package = package
            self.scores = scores
            self.entries = list(package.all_answers())  # [(case, answer)]
            self.index = None
            self.grader = None
            self.item_vars = []
            self._build()
            self.refresh_progress_list()
            self.goto(self.first_ungraded(), force=True)
            root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- layout ----

        def _build(self):
            self.root.title(
                "LLM Medical Cases - Answer Scorer - {}".format(
                    os.path.basename(self.package.zip_path)
                )
            )
            header = tk.Label(
                self.root,
                text="The answers are anonymized; the letters are shuffled for every "
                "case, so answer A on one case is NOT the same AI as on another.",
                anchor="w",
            )
            header.pack(fill="x", padx=8, pady=(8, 0))

            pane = tk.PanedWindow(self.root, orient="horizontal", sashrelief="raised")
            pane.pack(fill="both", expand=True, padx=8, pady=8)

            left = tk.Frame(pane)
            tk.Label(left, text="Answers to grade:", anchor="w").pack(fill="x")
            self.progress_list = tk.Listbox(left, width=26, exportselection=False)
            self.progress_list.pack(fill="both", expand=True, pady=4)
            self.progress_list.bind("<<ListboxSelect>>", self.on_pick)
            self.progress_label = tk.Label(left, text="", anchor="w", justify="left")
            self.progress_label.pack(fill="x")
            pane.add(left, minsize=210)

            right = tk.Frame(pane)
            self.case_header = tk.Label(
                right, text="", anchor="w", font=("TkDefaultFont", 10, "bold")
            )
            self.case_header.pack(fill="x")
            self.case_text = scrolledtext.ScrolledText(
                right, height=7, wrap="word", state="disabled"
            )
            self.case_text.pack(fill="both", expand=False, pady=(2, 6))
            answer_row = tk.Frame(right)
            answer_row.pack(fill="x")
            self.answer_header = tk.Label(
                answer_row, text="", anchor="w", font=("TkDefaultFont", 10, "bold")
            )
            self.answer_header.pack(side="left")
            self.images_button = tk.Button(
                answer_row, text="Open the images", command=self.open_images
            )
            self.formatted_button = tk.Button(
                answer_row, text="Read formatted (web browser)",
                command=self.open_formatted,
            )
            self.formatted_button.pack(side="right", padx=(0, 6))
            self.answer_text = scrolledtext.ScrolledText(
                right, height=9, wrap="word", state="disabled"
            )
            self.answer_text.pack(fill="both", expand=True, pady=(2, 6))

            self.grading_holder = tk.Frame(right)
            self.grading_holder.pack(fill="x")

            bottom = tk.Frame(right)
            bottom.pack(fill="x", pady=(6, 0))
            tk.Label(bottom, text="Comment for the investigator (optional):").pack(anchor="w")
            self.comment_entry = tk.Entry(bottom)
            self.comment_entry.pack(fill="x", pady=(0, 6))
            buttons = tk.Frame(bottom)
            buttons.pack(fill="x")
            self.verdict_label = tk.Label(buttons, text="", anchor="w")
            self.verdict_label.pack(side="left", fill="x", expand=True)
            self.save_button = tk.Button(
                buttons, text="Save grade + next", command=self.save_and_next
            )
            self.save_button.pack(side="right")
            pane.add(right)

        # ---- navigation ----

        def first_ungraded(self):
            for i, (case, answer) in enumerate(self.entries):
                if self.scores.get(case["case_id"], answer["label"]) is None:
                    return i
            return 0 if self.entries else None

        def refresh_progress_list(self):
            self.progress_list.delete(0, "end")
            graded = 0
            tallies = {0: 0, 1: 0, 2: 0}
            for case, answer in self.entries:
                record = self.scores.get(case["case_id"], answer["label"])
                if record is None:
                    mark = "-"
                else:
                    graded += 1
                    tallies[record["score"]] += 1
                    mark = "score {}".format(record["score"])
                self.progress_list.insert(
                    "end", "{}  {}   {}".format(case["case_id"], answer["label"], mark)
                )
            text = "Graded {} of {}.".format(graded, len(self.entries))
            if graded:
                text += "\n{}x 0, {}x 1, {}x 2".format(tallies[0], tallies[1], tallies[2])
            if graded == len(self.entries):
                text += "\nAll done! Email\n{}\nback to the PI.".format(self.scores.path)
            self.progress_label.configure(text=text)

        def on_pick(self, _event):
            selection = self.progress_list.curselection()
            if selection and selection[0] != self.index:
                self.goto(selection[0])

        def goto(self, index, force=False):
            if index is None:
                return
            if not force and not self.confirm_leaving():
                self.progress_list.selection_clear(0, "end")
                if self.index is not None:
                    self.progress_list.selection_set(self.index)
                return
            self.index = index
            case, answer = self.entries[index]
            previous = self.scores.get(case["case_id"], answer["label"])
            self.grader = AnswerGrader(case["rubric"], previous)
            self.case_header.configure(text="Case {}".format(case["case_id"]))
            self._set_text(self.case_text, case["case_text"])
            self.answer_header.configure(text="Answer {}:".format(answer["label"]))
            self._set_text(self.answer_text, answer.get("response_text", "") or "(no text)")
            if answer.get("images"):
                self.images_button.configure(
                    text="Open the {} image{}".format(
                        len(answer["images"]), "" if len(answer["images"]) == 1 else "s"
                    )
                )
                self.images_button.pack(side="right")
            else:
                self.images_button.pack_forget()
            self.comment_entry.delete(0, "end")
            self.comment_entry.insert(0, self.grader.comment)
            self.build_grading_rows(case)
            self.progress_list.selection_clear(0, "end")
            self.progress_list.selection_set(index)
            self.progress_list.see(index)
            self.refresh_verdict()

        def confirm_leaving(self):
            """Warn when leaving an answer with unsaved judgments."""
            if self.grader is None or self.index is None:
                return True
            case, answer = self.entries[self.index]
            previous = self.scores.get(case["case_id"], answer["label"])
            touched = any(r is not None for r in self.grader.results) or (
                self.comment_entry.get().strip() != (self.grader.comment or "")
            )
            if previous is None and touched:
                return messagebox.askyesno(
                    "Leave without saving?",
                    "This answer's grade has not been saved.\n\nLeave anyway?",
                    parent=self.root,
                )
            return True

        # ---- grading widgets ----

        def build_grading_rows(self, case):
            for child in self.grading_holder.winfo_children():
                child.destroy()
            self.item_vars = []
            tk.Label(
                self.grading_holder,
                text="Rubric - scores: 0 = item missed or unnecessary risk, "
                "1 = covered but poor approach, 2 = covered and sound.",
                anchor="w",
                justify="left",
                wraplength=640,
            ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
            for i, item in enumerate(case["rubric"]):
                var = tk.StringVar(
                    value="" if self.grader.results[i] is None
                    else ("yes" if self.grader.results[i] else "no")
                )
                self.item_vars.append(var)
                tk.Label(
                    self.grading_holder, text="{}. {}".format(i + 1, item),
                    anchor="w", justify="left", wraplength=460,
                ).grid(row=i + 1, column=0, sticky="w")
                tk.Radiobutton(
                    self.grading_holder, text="Covered", variable=var, value="yes",
                    tristatevalue="__none__",
                    command=lambda i=i: self.on_item(i, True),
                ).grid(row=i + 1, column=1, sticky="w")
                tk.Radiobutton(
                    self.grading_holder, text="Missed", variable=var, value="no",
                    tristatevalue="__none__",
                    command=lambda i=i: self.on_item(i, False),
                ).grid(row=i + 1, column=2, sticky="w")
                tk.Button(
                    self.grading_holder, text="Flag...",
                    command=lambda i=i: self.flag_item(i),
                ).grid(row=i + 1, column=3, sticky="w", padx=(6, 0))
            base = len(case["rubric"]) + 1
            self.risk_var = tk.StringVar(
                value="" if self.grader.unnecessary_risk is None
                else ("yes" if self.grader.unnecessary_risk else "no")
            )
            self.risk_label = tk.Label(
                self.grading_holder,
                text="Did the answer take any unnecessary risk with the patient?",
                anchor="w", justify="left", wraplength=460,
            )
            self.risk_label.grid(row=base, column=0, sticky="w", pady=(6, 0))
            self.risk_yes = tk.Radiobutton(
                self.grading_holder, text="Yes", variable=self.risk_var, value="yes",
                tristatevalue="__none__",
                command=lambda: self.on_risk(True),
            )
            self.risk_no = tk.Radiobutton(
                self.grading_holder, text="No", variable=self.risk_var, value="no",
                tristatevalue="__none__",
                command=lambda: self.on_risk(False),
            )
            self.risk_yes.grid(row=base, column=1, sticky="w", pady=(6, 0))
            self.risk_no.grid(row=base, column=2, sticky="w", pady=(6, 0))

            self.poor_var = tk.StringVar(
                value="" if self.grader.poor_approach is None
                else ("yes" if self.grader.poor_approach else "no")
            )
            self.poor_label = tk.Label(
                self.grading_holder,
                text="Was the approach poor, even though everything was covered?",
                anchor="w", justify="left", wraplength=460,
            )
            self.poor_label.grid(row=base + 1, column=0, sticky="w")
            self.poor_yes = tk.Radiobutton(
                self.grading_holder, text="Yes", variable=self.poor_var, value="yes",
                tristatevalue="__none__",
                command=lambda: self.on_poor(True),
            )
            self.poor_no = tk.Radiobutton(
                self.grading_holder, text="No", variable=self.poor_var, value="no",
                tristatevalue="__none__",
                command=lambda: self.on_poor(False),
            )
            self.poor_yes.grid(row=base + 1, column=1, sticky="w")
            self.poor_no.grid(row=base + 1, column=2, sticky="w")
            self.grading_holder.columnconfigure(0, weight=1)
            self.refresh_enables()

        def flag_item(self, index):
            """Tell the PI a rubric item seems too difficult or wrong. The
            flag travels back inside the scores file; only the PI can
            actually change the rubric (every scorer must grade against
            the same one)."""
            case, _answer = self.entries[self.index]
            item = case["rubric"][index]
            note = simpledialog.askstring(
                "Flag rubric item {}".format(index + 1),
                "Item: {}\n\nWhat seems wrong with it (too difficult, factually "
                "wrong, ambiguous...)? Your note goes to the investigator with "
                "your scores file:".format(item),
                parent=self.root,
            )
            if note is None or not note.strip():
                return
            self.scores.add_flag(case["case_id"], index, item, note.strip())
            messagebox.showinfo(
                "Flagged",
                "Noted. Keep grading against the CURRENT wording for now - if "
                "the investigator changes the rubric, the affected answers "
                "will automatically come back for re-grading.",
                parent=self.root,
            )

        def on_item(self, index, met):
            self.grader.set_item(index, met)
            self.refresh_enables()
            self.refresh_verdict()

        def on_risk(self, value):
            self.grader.unnecessary_risk = value
            self.refresh_enables()
            self.refresh_verdict()

        def on_poor(self, value):
            self.grader.poor_approach = value
            self.refresh_verdict()

        def refresh_enables(self):
            risk_state = "normal" if self.grader.risk_applies() else "disabled"
            for widget in (self.risk_label, self.risk_yes, self.risk_no):
                widget.configure(state=risk_state)
            if not self.grader.risk_applies():
                self.risk_var.set("")
                self.grader.unnecessary_risk = None
            poor_state = "normal" if self.grader.poor_applies() else "disabled"
            for widget in (self.poor_label, self.poor_yes, self.poor_no):
                widget.configure(state=poor_state)
            if not self.grader.poor_applies():
                self.poor_var.set("")
                self.grader.poor_approach = None

        def refresh_verdict(self):
            self.verdict_label.configure(text=self.grader.explanation())
            self.save_button.configure(
                state="normal" if self.grader.complete() else "disabled"
            )

        # ---- actions ----

        def open_formatted(self):
            """The current answer, rendered readable in the web browser."""
            if self.index is None:
                return
            case, answer = self.entries[self.index]
            out_dir = "formatted_answers"
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(
                out_dir, "{}_{}.html".format(case["case_id"], answer["label"])
            )
            document = markdown_to_html(
                answer.get("response_text", "") or "(no text)",
                title="Case {} - answer {}".format(case["case_id"], answer["label"]),
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(document)
            webbrowser.open("file:///" + os.path.abspath(path).replace(os.sep, "/"))

        def open_images(self):
            case, answer = self.entries[self.index]
            out_dir = images_dir_for(os.path.basename(self.package.zip_path))
            paths = self.package.extract_images(answer, out_dir)
            if not paths:
                messagebox.showinfo(
                    "No images", "No image files could be read from the package.",
                    parent=self.root,
                )
                return
            for path in paths:
                full = os.path.abspath(path)
                try:
                    os.startfile(full)  # Windows default image viewer
                except AttributeError:
                    import webbrowser

                    webbrowser.open("file://" + full)

        def save_and_next(self):
            if not self.grader.complete():
                return
            case, answer = self.entries[self.index]
            results, risk, poor = self.grader.normalized()
            self.scores.upsert(
                case["case_id"], answer["label"], results, risk, poor,
                self.comment_entry.get().strip(),
                rubric_version=case.get("rubric_version", 1),
            )
            self.refresh_progress_list()
            for offset in range(1, len(self.entries) + 1):
                candidate = (self.index + offset) % len(self.entries)
                entry_case, entry_answer = self.entries[candidate]
                if self.scores.get(entry_case["case_id"], entry_answer["label"]) is None:
                    self.goto(candidate, force=True)
                    return
            self.goto(self.index, force=True)  # everything graded; stay put
            messagebox.showinfo(
                "All done",
                "All {} answers are graded.\n\nEmail {} back to the principal "
                "investigator.".format(len(self.entries), self.scores.path),
                parent=self.root,
            )

        def on_close(self):
            if self.confirm_leaving():
                self.root.destroy()

        @staticmethod
        def _set_text(widget, content):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", content)
            widget.configure(state="disabled")

    def choose_package_folder(root):
        """Ask where the scoring package zip lives; returns the folder or
        None if the scorer gave up. Keeps asking until a folder with a
        package in it is chosen."""
        start_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
        while True:
            folder = filedialog.askdirectory(
                parent=root,
                title="Where is the scoring package? Choose the folder that "
                "contains the .zip you were emailed",
                initialdir=start_dir,
            )
            if not folder:
                # Cancelled: fall back to the program's own folder if the
                # zip happens to be there, otherwise bow out politely.
                if find_package_zips(start_dir):
                    return start_dir
                messagebox.showinfo(
                    "No folder chosen",
                    "To grade answers, start the program again and choose the "
                    "folder where you saved the zip you were emailed.",
                )
                return None
            if find_package_zips(folder):
                return folder
            if not messagebox.askretrycancel(
                "No package there",
                "No scoring package (.zip) was found in:\n{}\n\nChoose the "
                "folder where you saved the zip you were emailed.".format(folder),
            ):
                return None
            start_dir = folder

    def pick_zip(root, zips):
        if len(zips) == 1:
            return zips[0]
        dialog = tk.Toplevel(root)
        dialog.title("Which package?")
        dialog.grab_set()
        tk.Label(
            dialog, text="More than one scoring package was found in this folder.\n"
            "Which one do you want to score?",
        ).pack(padx=12, pady=(12, 4))
        box = tk.Listbox(dialog, width=50, height=min(10, len(zips)))
        for name in zips:
            box.insert("end", name)
        box.selection_set(0)
        box.pack(padx=12, pady=4)
        chosen = []

        def accept(_event=None):
            if box.curselection():
                chosen.append(zips[box.curselection()[0]])
            dialog.destroy()

        box.bind("<Double-Button-1>", accept)
        tk.Button(dialog, text="OK", width=10, command=accept).pack(pady=(4, 12))
        dialog.wait_window()
        return chosen[0] if chosen else None


def main():
    if not TK_AVAILABLE:
        print(
            "This program needs the window system that normally ships with "
            "Python (Tkinter). Please reinstall Python from python.org with "
            "the default options."
        )
        return 1
    root = tk.Tk()
    root.title("LLM Medical Cases - Answer Scorer")
    root.geometry("980x720")
    root.withdraw()
    folder = choose_package_folder(root)
    if folder is None:
        root.destroy()
        return 0
    # Work inside the chosen folder: the grades file and any extracted
    # images are saved next to the zip they belong to.
    os.chdir(folder)
    zips = find_package_zips()
    zip_name = pick_zip(root, zips)
    if not zip_name:
        root.destroy()
        return 0
    try:
        package = Package.load(zip_name)
        scores = ScoresStore.load_or_create(scores_path_for(zip_name), package.manifest)
    except PackageError as error:
        messagebox.showerror("Cannot open the package", str(error))
        root.destroy()
        return 1
    # Rubric fixes from the investigator: any rubric_update_*.json saved
    # next to the zip is applied now, re-queueing only the grades the
    # change affects.
    updates, update_problems = load_rubric_updates(find_rubric_updates())
    for problem in update_problems:
        print("Skipping rubric update file - " + problem)
    if updates:
        update_messages = apply_rubric_updates(package, scores, updates)
        if update_messages:
            messagebox.showinfo(
                "Rubric update applied",
                "The investigator changed one or more rubrics:\n\n{}".format(
                    "\n".join(update_messages)
                ),
            )
    if not scores.scorer:
        name = simpledialog.askstring(
            "Your name", "Your name or initials (stored with your grades):", parent=root
        )
        if name and name.strip():
            scores.scorer = name.strip()
            scores.save()
    ScorerApp(root, package, scores)
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
