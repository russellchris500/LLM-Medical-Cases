#!/usr/bin/env python3
"""Program 2 of the LLM Medical Cases evaluation framework: the Case Merger.

The principal investigator (PI) uses this tool to combine the case files
emailed in by providers (provider_NNN_cases.json, produced by Program 1's
case_editor.py) into a single master database, master_cases.json, which the
later programs (LLM runner, answer scorer, LLM ranker) read.

Everything is menu-driven - just run:  python3 merge_cases.py
The program finds the provider files in the current folder, lets you choose
which to merge, and asks about anything that needs a decision.

Each incoming file is validated before anything is merged. Because case IDs
embed the provider number (PPP-CCC), files from different providers can never
collide; a case ID already present in the master can only come from the same
provider re-sending an updated file. For those, the program shows both
versions and asks which to keep (suggesting the newer one).

Requires only the Python 3 standard library.
"""

import json
import os
import re
import tempfile
import time
from datetime import datetime

from case_editor import FORMAT_VERSION, CaseStore, CaseStoreError, now_iso

MASTER_FILENAME = "master_cases.json"


def parse_timestamp(value):
    """Parse an ISO-8601 timestamp; returns None if missing or malformed."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def same_content(a, b):
    return a["case_text"] == b["case_text"] and a["rubric"] == b["rubric"]


class MasterStore:
    """The PI's master case database: every provider's cases, keyed by case ID."""

    def __init__(self, path):
        self.path = path
        self.cases = {}  # case_id -> case dict (includes provider_number)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise CaseStoreError("{} is not valid JSON: {}".format(path, e))
        if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
            raise CaseStoreError(
                "{} is not a version-{} master case file.".format(path, FORMAT_VERSION)
            )
        store = cls(path)
        for raw in data.get("cases", []):
            provider_number = raw.get("provider_number") if isinstance(raw, dict) else None
            if not isinstance(provider_number, int) or provider_number <= 0:
                raise CaseStoreError(
                    "{} contains a case with an invalid provider_number.".format(path)
                )
            case = CaseStore._validate_case(raw, provider_number)
            case["provider_number"] = provider_number
            if case["case_id"] in store.cases:
                raise CaseStoreError(
                    "{} contains duplicate case ID {}.".format(path, case["case_id"])
                )
            store.cases[case["case_id"]] = case
        return store

    @classmethod
    def load_or_create(cls, path):
        if os.path.exists(path):
            return cls.load(path)
        return cls(path)

    def save(self):
        ordered = sorted(
            self.cases.values(), key=lambda c: (c["provider_number"], c["case_number"])
        )
        data = {"format_version": FORMAT_VERSION, "cases": ordered}
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, self.path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def provider_case_ids(self, provider_number):
        return {
            case_id
            for case_id, case in self.cases.items()
            if case["provider_number"] == provider_number
        }

    def merge_provider(self, provider_store, on_conflict=None, on_missing=None):
        """Merge one provider's CaseStore into the master.

        on_conflict(existing, incoming) is called when a case ID exists in
        both with different content; return True to take the incoming
        version. If omitted, the version with the newer updated_at wins (ties
        keep the existing version).

        on_missing(existing) is called for master cases from this provider
        that are absent from the incoming file (i.e. the provider deleted
        them); return True to remove the case from the master. If omitted,
        missing cases are kept.

        Returns a report dict of case-ID lists by outcome.
        """
        report = {
            "added": [],
            "updated": [],
            "unchanged": [],
            "kept_existing": [],
            "removed": [],
            "missing_kept": [],
        }
        for case in provider_store.cases:
            incoming = dict(case)
            incoming["provider_number"] = provider_store.provider_number
            existing = self.cases.get(incoming["case_id"])
            if existing is None:
                self.cases[incoming["case_id"]] = incoming
                report["added"].append(incoming["case_id"])
            elif same_content(existing, incoming):
                report["unchanged"].append(incoming["case_id"])
            else:
                if on_conflict is not None:
                    take_incoming = on_conflict(existing, incoming)
                else:
                    old = parse_timestamp(existing.get("updated_at"))
                    new = parse_timestamp(incoming.get("updated_at"))
                    take_incoming = old is not None and new is not None and new > old
                if take_incoming:
                    # Rubric versions must only ever go up in the master,
                    # even if the provider's file never saw the PI's bumps.
                    if incoming["rubric"] != existing["rubric"]:
                        incoming["rubric_version"] = max(
                            int(incoming.get("rubric_version", 1)),
                            int(existing.get("rubric_version", 1)) + 1,
                        )
                    else:
                        incoming["rubric_version"] = max(
                            int(incoming.get("rubric_version", 1)),
                            int(existing.get("rubric_version", 1)),
                        )
                    self.cases[incoming["case_id"]] = incoming
                    report["updated"].append(incoming["case_id"])
                else:
                    report["kept_existing"].append(incoming["case_id"])

        incoming_ids = {c["case_id"] for c in provider_store.cases}
        for case_id in sorted(self.provider_case_ids(provider_store.provider_number)):
            if case_id in incoming_ids:
                continue
            if on_missing is not None and on_missing(self.cases[case_id]):
                del self.cases[case_id]
                report["removed"].append(case_id)
            else:
                report["missing_kept"].append(case_id)
        return report

    def edit_rubric(self, case_id, new_rubric, ops=None, reason=""):
        """Apply a PI rubric fix: bump rubric_version and keep the full
        before/after in rubric_history so the change is auditable and can
        be sent to scorers as a rubric-update file.

        ops describes what happened to each OLD item index, so scorers can
        invalidate only the affected grades:
          {"removed": [old indexes], "reworded": [old indexes],
           "added": [new indexes]}
        Pass ops=None when the mapping is unknown; scorers then re-grade
        every answer on the case (the safe fallback).
        """
        case = self.cases.get(case_id)
        if case is None:
            raise CaseStoreError("No case {} in the master database.".format(case_id))
        cleaned = []
        for item in new_rubric:
            if not isinstance(item, str) or not item.strip():
                raise CaseStoreError("Rubric items must be non-empty text.")
            cleaned.append(item.strip())
        if not cleaned:
            raise CaseStoreError("The rubric must keep at least one item.")
        if cleaned == case["rubric"]:
            return case  # nothing changed
        old_version = case.get("rubric_version", 1)
        old_version = old_version if isinstance(old_version, int) and old_version >= 1 else 1
        entry = {
            "from_version": old_version,
            "to_version": old_version + 1,
            "old_rubric": list(case["rubric"]),
            "new_rubric": list(cleaned),
            "ops": ops,
            "reason": reason,
            "edited_at": now_iso(),
        }
        case["rubric"] = cleaned
        case["rubric_version"] = old_version + 1
        case.setdefault("rubric_history", []).append(entry)
        case["updated_at"] = now_iso()
        return case


def rubric_update_payload(master, case_ids):
    """The small emailable rubric-update file: for each edited case, the
    current rubric, its version, and the latest change's item mapping.
    Contains no answers and nothing that could unblind a scorer."""
    cases = []
    for case_id in sorted(case_ids):
        case = master.cases[case_id]
        history = case.get("rubric_history") or []
        latest = history[-1] if history else None
        cases.append({
            "case_id": case_id,
            "rubric": list(case["rubric"]),
            "rubric_version": case.get("rubric_version", 1),
            "from_version": latest["from_version"] if latest else None,
            "ops": latest["ops"] if latest else None,
            "reason": (latest.get("reason") or "") if latest else "",
        })
    return {
        "format_version": FORMAT_VERSION,
        "kind": "rubric_update",
        "created_at": now_iso(),
        "cases": cases,
    }


def read_rubric_flags(paths):
    """Collect rubric_flags entries from scorer scores_*.json files.
    Returns (flags, problems); each flag gains the scorer name and file."""
    flags = []
    problems = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as error:
            problems.append("{}: {}".format(os.path.basename(path), error))
            continue
        if not isinstance(data, dict):
            problems.append("{}: not a scores file".format(os.path.basename(path)))
            continue
        for flag in data.get("rubric_flags", []):
            if isinstance(flag, dict) and isinstance(flag.get("case_id"), str):
                enriched = dict(flag)
                enriched["scorer"] = data.get("scorer", "")
                enriched["file"] = os.path.basename(path)
                flags.append(enriched)
    return flags, problems


# ---------- window interface ----------


def preview(case, limit=68):
    flat = " ".join(case["case_text"].split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def find_provider_files(folder="."):
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(
        name for name in names if re.fullmatch(r"provider_\d{3,}_cases\.json", name)
    )


class MergeApp:
    def __init__(self, root, master, gui, files_folder="."):
        self.root = root
        self.master = master
        self.gui = gui
        self.files_folder = files_folder
        tk = gui.tk
        self.summary = tk.Label(root, text="", anchor="w")
        self.summary.pack(fill="x", padx=8, pady=(8, 0))

        body = tk.Frame(root)
        body.pack(fill="both", expand=True, padx=8, pady=8)
        left = tk.Frame(body)
        left.pack(side="left", fill="y")
        self.folder_label = tk.Label(left, text="", anchor="w", wraplength=280, justify="left")
        self.folder_label.pack(fill="x")
        self.file_list = tk.Listbox(left, selectmode="extended", width=36, exportselection=False)
        self.file_list.pack(fill="y", expand=True, pady=4)
        row = tk.Frame(left)
        row.pack(fill="x")
        tk.Button(row, text="Change folder...", command=self.change_folder).pack(side="left")
        tk.Button(row, text="Refresh", command=self.refresh_files).pack(side="left", padx=4)
        row2 = tk.Frame(left)
        row2.pack(fill="x", pady=(4, 0))
        tk.Button(row2, text="Merge selected", command=self.merge_selected).pack(side="left")
        tk.Button(row2, text="Merge all", command=self.merge_all).pack(side="left", padx=4)
        tk.Button(left, text="View the master database", command=self.view_master).pack(
            fill="x", pady=(8, 0)
        )
        tk.Label(left, text="Rubric fixes:", anchor="w").pack(fill="x", pady=(10, 0))
        tk.Button(
            left, text="Review rubric flags from scorers...",
            command=self.review_rubric_flags,
        ).pack(fill="x", pady=(2, 0))
        tk.Button(
            left, text="Edit a case's rubric...", command=self.edit_rubric_prompt
        ).pack(fill="x", pady=(2, 0))
        self.update_button = tk.Button(
            left, text="Save a rubric update file (0 changes)",
            command=self.save_rubric_update, state="disabled",
        )
        self.update_button.pack(fill="x", pady=(2, 0))
        # Cases edited this session, for the rubric-update file.
        self.edited_case_ids = set()

        right = tk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(right, text="What happened:", anchor="w").pack(fill="x")
        import gui_common
        self.log = gui_common.LogBox(right, height=20).pack(fill="both", expand=True, pady=4)

        self.refresh_files()
        self.update_summary()

    def update_summary(self):
        providers = {c["provider_number"] for c in self.master.cases.values()}
        self.summary.configure(
            text="Master database {}: {} case{} from {} provider{}.".format(
                MASTER_FILENAME,
                len(self.master.cases),
                "" if len(self.master.cases) == 1 else "s",
                len(providers),
                "" if len(providers) == 1 else "s",
            )
        )

    def refresh_files(self):
        self.folder_label.configure(
            text="Provider case files in:\n{}".format(os.path.abspath(self.files_folder))
        )
        self.file_list.delete(0, "end")
        for name in find_provider_files(self.files_folder):
            self.file_list.insert("end", name)

    def change_folder(self):
        from tkinter import filedialog

        folder = filedialog.askdirectory(
            parent=self.root,
            title="Where are the provider case files the providers emailed you?",
            initialdir=os.path.abspath(self.files_folder),
        )
        if folder:
            self.files_folder = folder
            self.refresh_files()
            if not find_provider_files(folder):
                self.log.log(
                    "No provider case files (provider_NNN_cases.json) in {} - "
                    "save the emailed files there and click Refresh.".format(folder)
                )

    def merge_selected(self):
        names = [self.file_list.get(i) for i in self.file_list.curselection()]
        if not names:
            self.gui.messagebox.showinfo(
                "Nothing selected",
                "Click the provider files you want to merge first (Ctrl-click "
                "selects several), or use Merge all.",
                parent=self.root,
            )
            return
        self.merge_files(names)

    def merge_all(self):
        names = find_provider_files(self.files_folder)
        if not names:
            self.gui.messagebox.showinfo(
                "No files found",
                "No provider case files (provider_NNN_cases.json) were found in\n"
                "{}\n\nUse 'Change folder...' to point at the folder where you "
                "saved the emailed files.".format(os.path.abspath(self.files_folder)),
                parent=self.root,
            )
            return
        self.merge_files(names)

    def on_conflict(self, existing, incoming):
        import gui_common

        old = parse_timestamp(existing.get("updated_at"))
        new = parse_timestamp(incoming.get("updated_at"))
        newer = "incoming" if (old is not None and new is not None and new > old) else "master"
        message = (
            "Case {} differs between the master and the incoming file.\n\n"
            "In the master:  edited {}  ({} rubric items)\n  {}\n\n"
            "Incoming:  edited {}  ({} rubric items)\n  {}\n\n"
            "The {} version is newer.".format(
                existing["case_id"],
                existing.get("updated_at"), len(existing["rubric"]), preview(existing),
                incoming.get("updated_at"), len(incoming["rubric"]), preview(incoming),
                "incoming" if newer == "incoming" else "master's",
            )
        )
        choice = gui_common.ask_choice(
            self.root,
            "Which version should be kept?",
            message,
            [("incoming", "Use the incoming version"), ("master", "Keep the master's version")],
        )
        return choice == "incoming"

    def on_missing(self, existing):
        return self.gui.messagebox.askyesno(
            "Case deleted by the provider?",
            "Case {} is in the master but not in the incoming file (the provider "
            "may have deleted it).\n\n  {}\n\nRemove it from the master too?\n"
            "(Choose No to keep it.)".format(existing["case_id"], preview(existing)),
            parent=self.root,
        )

    def merge_files(self, names):
        stores = []
        for name in names:
            try:
                stores.append(CaseStore.load(os.path.join(self.files_folder, name)))
            except (CaseStoreError, OSError) as error:
                self.gui.messagebox.showerror(
                    "Problem with {}".format(name),
                    "{}\n\nNothing was merged. Ask the provider to re-send the "
                    "file, then try again.".format(error),
                    parent=self.root,
                )
                return
        for name, store in zip(names, stores):
            report = self.master.merge_provider(
                store, on_conflict=self.on_conflict, on_missing=self.on_missing
            )
            self.log.log("Result for {} (provider {}):".format(name, store.provider_number))
            labels = [
                ("added", "added"),
                ("updated", "updated to the newer version"),
                ("unchanged", "already in the master, unchanged"),
                ("kept_existing", "kept the master's version"),
                ("removed", "removed (deleted by the provider)"),
                ("missing_kept", "missing from the incoming file but kept"),
            ]
            for key, label in labels:
                if report[key]:
                    self.log.log(
                        "   {:3d} {}: {}".format(len(report[key]), label, ", ".join(report[key]))
                    )
            if not any(report.values()):
                self.log.log("   nothing to do (the file has no cases)")
        try:
            self.master.save()
        except OSError as error:
            self.gui.messagebox.showerror("Could not save", str(error), parent=self.root)
            return
        self.log.log("Saved {} ({} cases total).".format(MASTER_FILENAME, len(self.master.cases)))
        self.log.log("")
        self.update_summary()

    # ---- rubric fixes ----

    def review_rubric_flags(self):
        import gui_common
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Choose the scores files the scorers emailed back (scores_*.json)",
            filetypes=[("Scores files", "*.json"), ("All files", "*.*")],
        )
        if not paths:
            return
        flags, problems = read_rubric_flags(paths)
        for problem in problems:
            self.log.log("Could not read {}".format(problem))
        if not flags:
            self.gui.messagebox.showinfo(
                "No flags",
                "No rubric flags were found in the chosen files - the scorers "
                "did not flag any rubric items.",
                parent=self.root,
            )
            return
        window = self.gui.tk.Toplevel(self.root)
        window.title("Rubric items flagged by scorers")
        window.geometry("900x420")
        tree = gui_common.make_table(
            window,
            [("case", "Case"), ("item", "Item #"), ("text", "Rubric item"),
             ("note", "Scorer's note"), ("scorer", "Scorer")],
            widths={"case": 80, "item": 60, "text": 330, "note": 260, "scorer": 90},
        )
        for i, flag in enumerate(flags):
            tree.insert("", "end", iid=str(i), values=(
                flag.get("case_id", ""),
                (flag.get("item_index", 0) or 0) + 1,
                flag.get("item_text", ""),
                flag.get("note", ""),
                flag.get("scorer", ""),
            ))
        tree.master.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        def edit_flagged():
            selection = tree.selection()
            if not selection:
                self.gui.messagebox.showinfo(
                    "Nothing selected", "Click a flag first.", parent=window
                )
                return
            case_id = flags[int(selection[0])].get("case_id", "")
            self.edit_rubric_dialog(case_id)

        row = self.gui.tk.Frame(window)
        row.pack(fill="x", padx=8, pady=8)
        self.gui.tk.Button(
            row, text="Edit this case's rubric...", command=edit_flagged
        ).pack(side="left")

    def edit_rubric_prompt(self):
        from tkinter import simpledialog

        case_id = simpledialog.askstring(
            "Which case?",
            "Case ID whose rubric needs fixing (e.g. 003-007):",
            parent=self.root,
        )
        if case_id:
            self.edit_rubric_dialog(case_id.strip())

    def edit_rubric_dialog(self, case_id):
        tk = self.gui.tk
        case = self.master.cases.get(case_id)
        if case is None:
            self.gui.messagebox.showerror(
                "Unknown case", "There is no case {} in the master database.".format(case_id),
                parent=self.root,
            )
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit the rubric of case {} (version {})".format(
            case_id, case.get("rubric_version", 1)
        ))
        dialog.geometry("760x560")
        tk.Label(
            dialog,
            text="Case text:\n{}".format(preview(case, 180)),
            anchor="w", justify="left", wraplength=720,
        ).pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(
            dialog,
            text="Edit an item's wording in its box; tick Remove to drop it. "
            "Add brand-new items at the bottom.",
            anchor="w", justify="left", wraplength=720,
        ).pack(fill="x", padx=10)

        holder = tk.Frame(dialog)
        holder.pack(fill="x", padx=10, pady=6)
        item_vars = []
        for index, item in enumerate(case["rubric"]):
            row = tk.Frame(holder)
            row.pack(fill="x", pady=2)
            tk.Label(row, text="{}.".format(index + 1), width=3, anchor="e").pack(side="left")
            text_var = tk.StringVar(value=item)
            tk.Entry(row, textvariable=text_var).pack(side="left", fill="x", expand=True)
            remove_var = tk.BooleanVar(value=False)
            tk.Checkbutton(row, text="Remove", variable=remove_var).pack(side="left", padx=4)
            item_vars.append((item, text_var, remove_var))

        tk.Label(dialog, text="New items to ADD (one per line):", anchor="w").pack(
            fill="x", padx=10
        )
        added_box = self.gui.tk.Text(dialog, height=4)
        added_box.pack(fill="x", padx=10, pady=(2, 6))
        reason_row = tk.Frame(dialog)
        reason_row.pack(fill="x", padx=10)
        tk.Label(reason_row, text="Why (kept in the audit trail):").pack(side="left")
        reason_var = tk.StringVar()
        tk.Entry(reason_row, textvariable=reason_var).pack(side="left", fill="x", expand=True)

        def apply_edit():
            new_rubric = []
            ops = {"removed": [], "reworded": [], "added": []}
            for old_index, (original, text_var, remove_var) in enumerate(item_vars):
                if remove_var.get():
                    ops["removed"].append(old_index)
                    continue
                text = text_var.get().strip()
                if text != original:
                    ops["reworded"].append(old_index)
                new_rubric.append(text)
            added = [line.strip() for line in added_box.get("1.0", "end").splitlines()
                     if line.strip()]
            for offset in range(len(added)):
                ops["added"].append(len(new_rubric) + offset)
            new_rubric.extend(added)
            if new_rubric == case["rubric"]:
                self.gui.messagebox.showinfo(
                    "No change", "The rubric is unchanged.", parent=dialog
                )
                return
            try:
                edited = self.master.edit_rubric(
                    case_id, new_rubric, ops=ops, reason=reason_var.get().strip()
                )
                self.master.save()
            except (CaseStoreError, OSError) as error:
                self.gui.messagebox.showerror("Cannot save", str(error), parent=dialog)
                return
            if case_id in self.edited_case_ids:
                # Two edits in one session: the update file can no longer
                # describe a single step, so scorers re-grade the whole case.
                history = edited.get("rubric_history") or []
                if history:
                    history[-1]["ops"] = None
            self.edited_case_ids.add(case_id)
            self.update_button.configure(
                state="normal",
                text="Save a rubric update file ({} change{})".format(
                    len(self.edited_case_ids),
                    "" if len(self.edited_case_ids) == 1 else "s",
                ),
            )
            self.log.log(
                "Rubric of case {} is now version {}. Remember to TELL PROVIDER {} "
                "so their own file gets the same fix.".format(
                    case_id, edited.get("rubric_version"), case.get("provider_number")
                )
            )
            self.log.log(
                "When done editing, click 'Save a rubric update file' and email "
                "it to every scorer who has this case."
            )
            self.update_summary()
            dialog.destroy()

        buttons = tk.Frame(dialog)
        buttons.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(buttons, text="Save the new rubric", command=apply_edit).pack(side="left")
        tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=6)

    def save_rubric_update(self):
        from tkinter import filedialog

        if not self.edited_case_ids:
            return
        payload = rubric_update_payload(self.master, self.edited_case_ids)
        default_name = "rubric_update_{}.json".format(time.strftime("%Y%m%d_%H%M%S"))
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save the rubric update file (email it to your scorers)",
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("Rubric update files", "*.json")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        self.log.log(
            "Saved {} covering {}: email it to every scorer together with "
            "score_answers.py's instructions - their program applies it and "
            "re-queues only the answers the change affects.".format(
                os.path.basename(path), ", ".join(sorted(self.edited_case_ids))
            )
        )

    def view_master(self):
        import gui_common

        window = self.gui.tk.Toplevel(self.root)
        window.title("Master database - {} cases".format(len(self.master.cases)))
        window.geometry("860x480")
        tree = gui_common.make_table(
            window,
            [("case", "Case"), ("items", "Rubric items"), ("version", "Rubric v"),
             ("text", "Case text")],
            widths={"case": 90, "items": 90, "version": 70, "text": 550},
        )
        ordered = sorted(
            self.master.cases.values(), key=lambda c: (c["provider_number"], c["case_number"])
        )
        for case in ordered:
            tree.insert("", "end", values=(
                case["case_id"], len(case["rubric"]),
                case.get("rubric_version", 1), preview(case, 100),
            ))
        tree.master.pack(fill="both", expand=True, padx=8, pady=8)


class _GuiModules:
    def __init__(self, tk, messagebox):
        self.tk = tk
        self.messagebox = messagebox


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import gui_common

    root = gui_common.make_root("Case Merger (for the PI)", 960, 620)
    try:
        master = MasterStore.load_or_create(MASTER_FILENAME)
    except (CaseStoreError, OSError) as error:
        gui_common.show_error("Cannot open the master database", str(error))
        root.destroy()
        return 1
    # The master database lives next to the programs; the emailed provider
    # files can be anywhere - ask where, unless they are already here.
    files_folder = "."
    if not find_provider_files("."):
        root.withdraw()
        chosen = filedialog.askdirectory(
            parent=root,
            title="Where are the provider case files the providers emailed you? "
            "(Cancel to choose later)",
            initialdir=os.getcwd(),
        )
        if chosen:
            files_folder = chosen
        root.deiconify()
    MergeApp(root, master, _GuiModules(tk, messagebox), files_folder=files_folder)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
