#!/usr/bin/env python3
"""Program 4 of the LLM Medical Cases evaluation framework: the Scoring
Package Builder.

The principal investigator (PI) uses this tool to bundle collected LLM
answers into a zip file that can be emailed to a scorer. The scorer's
selection of cases and models is made here, completely independently of
what was selected when the answers were collected.

Blinding: inside the package every answer is labeled only A, B, C... and
the labels are shuffled per case, so the scorer cannot tell which AI wrote
what or follow one AI's style across cases. The label-to-model key is
written to a separate file that stays with the PI and must NEVER be sent
to a scorer.

It is a window-based program - run it (or double-click
"Build Scoring Package.pyw") and work in the window.

Requires only the Python 3 standard library.
"""

import json
import os
import random
import re
import string
import tempfile
import zipfile

from case_editor import CaseStoreError, now_iso
from merge_cases import MASTER_FILENAME, MasterStore
from eval_common import (
    AnswersStore,
    CaseSetStore,
    OK_STATUSES,
    SET_NAME_RE,
    SelectionError,
    case_hash,
    parse_selection,
    sort_case_ids,
    split_case_id,
)

PACKAGES_DIR = "scoring_packages"
EMAIL_SIZE_LIMIT = 20 * 1024 * 1024  # stay under common 25 MB email caps

# Strings that give away which AI wrote an answer; found in answer text
# they are reported to the PI (never auto-redacted).
SELF_ID_STRINGS = [
    "ChatGPT", "OpenAI", "GPT-4", "GPT-5", "Claude", "Anthropic", "Gemini",
    "Google AI", "Grok", "xAI", "OpenEvidence", "UpToDate", "Doximity",
]

SCORER_README = """This package contains {num_cases} medical case(s) with anonymized AI answers.

Each case shows the case text, the grading rubric, and several answers
labeled A, B, C... The labels are shuffled for every case, so label A on
one case is NOT the same AI as label A on another case.

Please score the answers with the score_answers program, which reads this
zip file directly. Each answer is scored 0, 1, or 2:
  0 - any rubric item is missed, or the answer takes unnecessary risk
      with the patient
  1 - every rubric item is covered, but the approach is poor
  2 - every rubric item is covered and the approach is acceptable
"""


def random_suffix():
    rng = random.SystemRandom()
    return "".join(rng.choice(string.hexdigits.lower()) for _ in range(4))


def labels_for(count):
    return list(string.ascii_uppercase[:count])


def blind_cases(package_cases, answers_by_case):
    """Assign shuffled labels per case; returns (manifest_cases, key).

    manifest_cases: list for package.json (no model names anywhere).
    key: case_id -> label -> {model_id, model_reported} for the PI's file.
    """
    rng = random.SystemRandom()
    manifest = []
    key = {}
    for case in package_cases:
        case_id = case["case_id"]
        records = list(answers_by_case[case_id])
        rng.shuffle(records)
        labels = labels_for(len(records))
        entry_answers = []
        key[case_id] = {}
        for label, record in zip(labels, records):
            key[case_id][label] = {
                # The full scored identity (LLM + model name) and its parts.
                "model_id": record["model_id"],
                "llm_id": record.get("llm_id", record["model_id"]),
                "model_name": record.get("model_name", ""),
                "display_name": record.get("model_display_name", record["model_id"]),
                "model_reported": record.get("model_reported", ""),
            }
            entry_answers.append(
                {
                    "label": label,
                    "response_text": record.get("response_text", ""),
                    "images": [],  # filled in while the zip is written
                    "_source_images": record.get("images", []),
                }
            )
        manifest.append(
            {
                "case_id": case_id,
                "case_text": case["case_text"],
                "rubric": list(case["rubric"]),
                # Grades record which rubric version they were made under,
                # so a later rubric fix can invalidate exactly the grades
                # it affects (and the ranker can refuse mixed versions).
                "rubric_version": case.get("rubric_version", 1),
                "answers": entry_answers,
            }
        )
    return manifest, key


def blinded_image_name(case_id, label, source_path):
    base = os.path.basename(source_path)
    if base.endswith("_page.png"):
        suffix = "page.png"
    else:
        match = re.search(r"_(\d+)\.(\w+)$", base)
        suffix = "{}.{}".format(match.group(1), match.group(2)) if match else "001.png"
    return "images/{}_{}_{}".format(case_id, label, suffix)


def write_package_zip(zip_path, manifest_cases, package_meta, warn):
    """Write the zip atomically; returns the manifest actually written."""
    directory = os.path.dirname(os.path.abspath(zip_path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".zip.tmp")
    os.close(fd)
    try:
        cleaned_cases = []
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            for case in manifest_cases:
                cleaned_answers = []
                for answer in case["answers"]:
                    stored_names = []
                    for source in answer.get("_source_images", []):
                        if not os.path.exists(source):
                            warn(
                                "Image file {} is missing; the package will not "
                                "include it.".format(source)
                            )
                            continue
                        name = blinded_image_name(case["case_id"], answer["label"], source)
                        bundle.write(source, name)
                        stored_names.append(name)
                    cleaned_answers.append(
                        {
                            "label": answer["label"],
                            "response_text": answer["response_text"],
                            "images": stored_names,
                        }
                    )
                cleaned_cases.append(
                    {
                        "case_id": case["case_id"],
                        "case_text": case["case_text"],
                        "rubric": case["rubric"],
                        "rubric_version": case.get("rubric_version", 1),
                        "answers": cleaned_answers,
                    }
                )
            manifest = dict(package_meta)
            manifest["num_cases"] = len(cleaned_cases)
            manifest["cases"] = cleaned_cases
            bundle.writestr(
                "package.json",
                json.dumps(manifest, indent=2, ensure_ascii=False),
            )
            bundle.writestr(
                "README.txt", SCORER_README.format(num_cases=len(cleaned_cases))
            )
        os.replace(tmp_path, zip_path)
        return manifest
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def self_identification_warnings(manifest_cases):
    warnings = []
    for case in manifest_cases:
        for answer in case["answers"]:
            text = answer.get("response_text", "")
            hits = sorted({s for s in SELF_ID_STRINGS if s.lower() in text.lower()})
            if hits:
                warnings.append(
                    "Case {} answer {} mentions: {}".format(
                        case["case_id"], answer["label"], ", ".join(hits)
                    )
                )
    return warnings


def stale_package_warnings(master, out_dir=PACKAGES_DIR):
    """Which earlier packages contain a rubric that has since been edited.
    Scorers holding those zips need the PI's rubric update file."""
    warnings = []
    try:
        names = sorted(os.listdir(out_dir))
    except OSError:
        return warnings
    for name in names:
        if not name.endswith("_KEY_DO_NOT_SEND.json"):
            continue
        try:
            with open(os.path.join(out_dir, name), "r", encoding="utf-8") as f:
                key_data = json.load(f)
        except (OSError, ValueError):
            continue
        stale = []
        for case_id, packaged_version in (key_data.get("rubric_versions") or {}).items():
            case = master.cases.get(case_id)
            if case is not None and case.get("rubric_version", 1) > packaged_version:
                stale.append("{} (v{} -> v{})".format(
                    case_id, packaged_version, case.get("rubric_version", 1)
                ))
        if stale:
            warnings.append(
                "Package '{}' was built with older rubrics: {}. Send its scorer "
                "a rubric update file (Case Merger > Save a rubric update "
                "file).".format(
                    key_data.get("package_name", name), ", ".join(sorted(stale))
                )
            )
    return warnings


def build_package(name, case_ids, model_ids, master, answers, out_dir=PACKAGES_DIR, warn=print):
    """Assemble, blind, and write one package. Returns (zip_path, key_path)."""
    answers_by_case = {}
    for case_id in case_ids:
        records = []
        for model_id in model_ids:
            record = answers.get(case_id, model_id)
            if record is not None and record.get("status") in OK_STATUSES:
                records.append(record)
        answers_by_case[case_id] = records

    package_cases = [master.cases[c] for c in case_ids]
    manifest_cases, key = blind_cases(package_cases, answers_by_case)

    package_id = "pkg_{}_{}".format(
        now_iso().replace(":", "").replace("-", "")[:16], random_suffix()
    )
    meta = {
        "format_version": 1,
        "package_id": package_id,
        "package_name": name,
        "created_at": now_iso(),
        "labels_used": sorted({a["label"] for c in manifest_cases for a in c["answers"]}),
    }
    zip_path = os.path.join(out_dir, name + ".zip")
    manifest = write_package_zip(zip_path, manifest_cases, meta, warn)

    key_path = os.path.join(out_dir, name + "_KEY_DO_NOT_SEND.json")
    key_data = {
        "format_version": 1,
        "package_id": package_id,
        "package_name": name,
        "created_at": meta["created_at"],
        # Which rubric version each case was packaged with: the ranker
        # checks grades against this so mixed-rubric grades never rank.
        "rubric_versions": {
            case["case_id"]: case.get("rubric_version", 1) for case in manifest_cases
        },
        "key": key,
    }
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for warning in self_identification_warnings(manifest["cases"]):
        warn("Blinding note: " + warning)
    for warning in stale_package_warnings(master, out_dir):
        warn("Rubric note: " + warning)
    return zip_path, key_path


# ---------- window interface ----------


class PackageBuilderApp:
    def __init__(self, root, master, answers, case_sets, gui):
        self.root = root
        self.master = master
        self.answers = answers
        self.case_sets = case_sets
        self.gui = gui
        tk = gui.tk
        import gui_common

        usable = answers.answered_case_ids()
        self.usable_cases = [c for c in usable if c in master.cases]
        self.orphans = [c for c in usable if c not in master.cases]
        self.model_vars = {}

        top = tk.Label(
            root,
            text="Choose the cases and the AIs to send to a scorer. The zip is "
            "blinded; the matching *_KEY_DO_NOT_SEND.json stays with you.",
            anchor="w", justify="left",
        )
        top.pack(fill="x", padx=8, pady=(8, 0))

        body = tk.Frame(root)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Cases with usable answers (click to select, "
                 "Ctrl-click for several):", anchor="w").pack(fill="x")
        self.case_tree = gui_common.make_table(
            left, [("case", "Case"), ("count", "Answers"), ("text", "Case text")],
            widths={"case": 80, "count": 70, "text": 330},
        )
        self.case_tree.master.pack(fill="both", expand=True, pady=4)
        row = tk.Frame(left)
        row.pack(fill="x")
        tk.Button(row, text="Select all", command=self.select_all).pack(side="left")
        tk.Label(row, text="  or type a range:").pack(side="left")
        self.expression = tk.Entry(row, width=28)
        self.expression.pack(side="left", padx=4)
        tk.Button(row, text="Apply", command=self.apply_expression).pack(side="left")

        right = tk.Frame(body)
        right.pack(side="left", fill="y", padx=(10, 0))
        tk.Label(right, text="AIs to include:", anchor="w").pack(fill="x")
        for model_id in answers.model_ids():
            count = sum(
                1 for c in self.usable_cases
                if (answers.get(c, model_id) or {}).get("status") in OK_STATUSES
            )
            if count == 0:
                continue
            var = tk.BooleanVar(value=True)
            self.model_vars[model_id] = var
            display = model_id
            for c in self.usable_cases:
                record = answers.get(c, model_id)
                if record is not None:
                    display = record.get("model_display_name") or model_id
                    break
            tk.Checkbutton(
                right, text="{}  ({} answers)".format(display, count), variable=var,
                anchor="w",
            ).pack(fill="x")
        tk.Label(right, text="").pack()
        tk.Label(right, text="Package name (e.g. pilot20-drsmith):", anchor="w").pack(fill="x")
        self.name_entry = tk.Entry(right, width=28)
        self.name_entry.pack(fill="x", pady=(0, 6))
        tk.Button(right, text="Build the package", command=self.build).pack(fill="x")
        tk.Label(right, text="What happened:", anchor="w").pack(fill="x", pady=(10, 0))
        self.log = gui_common.LogBox(right, height=12).pack(fill="both", expand=True)

        if self.orphans:
            self.log.log(
                "Note: {} answered case(s) are no longer in master_cases.json and "
                "cannot be packaged: {}. Re-merge the provider file to bring "
                "them back.".format(len(self.orphans), ", ".join(self.orphans))
            )
        self.fill_cases()

    def fill_cases(self):
        per_case = {
            c: sum(
                1 for (cid, m), r in self.answers.answers.items()
                if cid == c and r.get("status") in OK_STATUSES
            )
            for c in self.usable_cases
        }
        for case_id in self.usable_cases:
            case = self.master.cases[case_id]
            flat = " ".join(case["case_text"].split())
            self.case_tree.insert(
                "", "end", iid=case_id,
                values=(case_id, per_case[case_id], flat[:60]),
            )

    def select_all(self):
        self.case_tree.selection_set(self.case_tree.get_children())

    def apply_expression(self):
        expression = self.expression.get().strip()
        if not expression:
            return
        available = {c: self.master.cases[c] for c in self.usable_cases}
        try:
            selected, warnings = parse_selection(expression, available)
        except SelectionError as error:
            self.gui.messagebox.showerror("Cannot read that", str(error), parent=self.root)
            return
        for warning in warnings:
            self.log.log("Note: " + warning)
        self.case_tree.selection_set([c for c in selected])

    def chosen_cases(self):
        return [iid for iid in self.case_tree.selection()]

    def chosen_models(self):
        return [m for m, var in self.model_vars.items() if var.get()]

    def build(self):
        import gui_common

        case_ids = sort_case_ids(self.chosen_cases())
        model_ids = self.chosen_models()
        if not case_ids:
            self.gui.messagebox.showinfo(
                "No cases selected", "Click the cases to include first (or Select all).",
                parent=self.root,
            )
            return
        if not model_ids:
            self.gui.messagebox.showinfo(
                "No AIs selected", "Tick at least one AI to include.", parent=self.root
            )
            return
        name = self.name_entry.get().strip()
        if not SET_NAME_RE.match(name or ""):
            self.gui.messagebox.showinfo(
                "Package name needed",
                "Give the package a name using only letters, digits, dots, dashes, "
                "and underscores (up to 40 characters).",
                parent=self.root,
            )
            return

        # Coverage check
        holes, failed, changed = [], [], []
        for case_id in case_ids:
            for model_id in model_ids:
                record = self.answers.get(case_id, model_id)
                if record is None:
                    holes.append((case_id, model_id))
                elif record.get("status") not in OK_STATUSES:
                    failed.append((case_id, model_id))
                elif record.get("case_sha256") != case_hash(self.master.cases[case_id]):
                    changed.append((case_id, model_id))
        for case_id, model_id in failed[:10]:
            self.log.log("Left out (failed when collected): {} x {}".format(case_id, model_id))
        for case_id, model_id in changed[:10]:
            self.log.log(
                "Note: the wording of case {} changed after {}'s answer was "
                "collected.".format(case_id, model_id)
            )
        incomplete = sorted({c for c, _ in holes + failed}, key=split_case_id)
        if incomplete:
            choice = gui_common.ask_choice(
                self.root,
                "Some cases are incomplete",
                "{} of the selected cases do not have an answer from every chosen "
                "AI (e.g. {}).".format(len(incomplete), ", ".join(incomplete[:6])),
                [
                    ("include", "Include them with the answers they have"),
                    ("exclude", "Exclude the incomplete cases"),
                    ("cancel", "Cancel"),
                ],
            )
            if choice in (None, "cancel"):
                return
            if choice == "exclude":
                case_ids = [c for c in case_ids if c not in set(incomplete)]
        case_ids = [
            c for c in case_ids
            if any(
                (self.answers.get(c, m) or {}).get("status") in OK_STATUSES
                for m in model_ids
            )
        ]
        if not case_ids:
            self.gui.messagebox.showinfo(
                "Nothing to package", "No selected case has a usable answer.",
                parent=self.root,
            )
            return

        zip_path = os.path.join(PACKAGES_DIR, name + ".zip")
        if os.path.exists(zip_path) and not self.gui.messagebox.askyesno(
            "Replace package?",
            "A package named {} already exists. Replace it?".format(name),
            parent=self.root,
        ):
            return

        try:
            zip_path, key_path = build_package(
                name, case_ids, model_ids, self.master, self.answers, warn=self.log.log
            )
        except (CaseStoreError, OSError) as error:
            self.gui.messagebox.showerror("Could not build it", str(error), parent=self.root)
            return

        size = os.path.getsize(zip_path)
        if size > EMAIL_SIZE_LIMIT and self.gui.messagebox.askyesno(
            "Large package",
            "The package is {:.1f} MB, which may be too large to email (most "
            "mail systems cap attachments around 25 MB).\n\nSplit it into "
            "parts of at most 20 MB?".format(size / (1024 * 1024)),
            parent=self.root,
        ):
            parts = split_package(zip_path, name, case_ids, model_ids,
                                  self.master, self.answers, key_path)
            for part in parts:
                self.log.log("Built {}  ({:.1f} MB)".format(
                    part, os.path.getsize(part) / (1024 * 1024)
                ))
        else:
            self.log.log("Built {}  ({:.1f} MB) - email this to the scorer.".format(
                zip_path, size / (1024 * 1024)
            ))
        self.log.log(
            "The *_KEY_DO_NOT_SEND.json file(s) in {} stay with you - NEVER "
            "send them to a scorer.".format(PACKAGES_DIR)
        )


def split_package(zip_path, name, case_ids, model_ids, master, answers, key_path):
    """Split by whole cases into parts of at most EMAIL_SIZE_LIMIT."""
    parts = []
    chunk = []
    number = 1
    for case_id in case_ids:
        chunk.append(case_id)
        part_name = "{}_part{}".format(name, number)
        part_path, _ = build_package(
            part_name, chunk, model_ids, master, answers, warn=lambda *_: None
        )
        if os.path.getsize(part_path) > EMAIL_SIZE_LIMIT and len(chunk) > 1:
            chunk.pop()
            part_path, _ = build_package(
                part_name, chunk, model_ids, master, answers, warn=lambda *_: None
            )
            parts.append(part_path)
            number += 1
            chunk = [case_id]
    if chunk:
        part_name = "{}_part{}".format(name, number)
        part_path, _ = build_package(
            part_name, chunk, model_ids, master, answers, warn=lambda *_: None
        )
        parts.append(part_path)
    os.unlink(zip_path)
    os.unlink(key_path)
    return parts


class _GuiModules:
    def __init__(self, tk, messagebox):
        self.tk = tk
        self.messagebox = messagebox


def main():
    import tkinter as tk
    from tkinter import messagebox
    import gui_common
    from eval_common import CaseSetStore

    root = gui_common.make_root("Scoring Package Builder (for the PI)", 1000, 620)
    try:
        answers = AnswersStore.load_or_create()
        case_sets = CaseSetStore.load_or_create()
        master = MasterStore.load(MASTER_FILENAME) if os.path.exists(MASTER_FILENAME) else None
    except (CaseStoreError, OSError) as error:
        gui_common.show_error("Cannot start", str(error))
        root.destroy()
        return 1
    if master is None:
        gui_common.show_error(
            "No master database",
            "No master case database ({}) was found in this folder.\n"
            "Run the Case Merger first.".format(MASTER_FILENAME),
        )
        root.destroy()
        return 1
    if not answers.answered_case_ids():
        gui_common.show_error(
            "No answers yet",
            "No usable answers were found (answers.json). Run the LLM Runner first.",
        )
        root.destroy()
        return 1
    PackageBuilderApp(root, master, answers, case_sets, _GuiModules(tk, messagebox))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
