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
from datetime import datetime

from case_editor import FORMAT_VERSION, CaseStore, CaseStoreError

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


# ---------- window interface ----------


def preview(case, limit=68):
    flat = " ".join(case["case_text"].split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def find_provider_files():
    return sorted(
        name for name in os.listdir(".") if re.fullmatch(r"provider_\d{3,}_cases\.json", name)
    )


class MergeApp:
    def __init__(self, root, master, gui):
        self.root = root
        self.master = master
        self.gui = gui
        tk = gui.tk
        self.summary = tk.Label(root, text="", anchor="w")
        self.summary.pack(fill="x", padx=8, pady=(8, 0))

        body = tk.Frame(root)
        body.pack(fill="both", expand=True, padx=8, pady=8)
        left = tk.Frame(body)
        left.pack(side="left", fill="y")
        tk.Label(left, text="Provider case files in this folder:", anchor="w").pack(fill="x")
        self.file_list = tk.Listbox(left, selectmode="extended", width=36, exportselection=False)
        self.file_list.pack(fill="y", expand=True, pady=4)
        row = tk.Frame(left)
        row.pack(fill="x")
        tk.Button(row, text="Refresh", command=self.refresh_files).pack(side="left")
        tk.Button(row, text="Merge selected", command=self.merge_selected).pack(side="left", padx=4)
        tk.Button(row, text="Merge all", command=self.merge_all).pack(side="left")
        tk.Button(left, text="View the master database", command=self.view_master).pack(
            fill="x", pady=(8, 0)
        )

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
        self.file_list.delete(0, "end")
        for name in find_provider_files():
            self.file_list.insert("end", name)

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
        names = find_provider_files()
        if not names:
            self.gui.messagebox.showinfo(
                "No files found",
                "No provider case files (provider_NNN_cases.json) were found in "
                "this folder. Copy the files the providers emailed you next to "
                "this program, then click Refresh.",
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
                stores.append(CaseStore.load(name))
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

    def view_master(self):
        import gui_common

        window = self.gui.tk.Toplevel(self.root)
        window.title("Master database - {} cases".format(len(self.master.cases)))
        window.geometry("860x480")
        tree = gui_common.make_table(
            window,
            [("case", "Case"), ("items", "Rubric items"), ("text", "Case text")],
            widths={"case": 90, "items": 90, "text": 620},
        )
        ordered = sorted(
            self.master.cases.values(), key=lambda c: (c["provider_number"], c["case_number"])
        )
        for case in ordered:
            tree.insert("", "end", values=(case["case_id"], len(case["rubric"]), preview(case, 100)))
        tree.master.pack(fill="both", expand=True, padx=8, pady=8)


class _GuiModules:
    def __init__(self, tk, messagebox):
        self.tk = tk
        self.messagebox = messagebox


def main():
    import tkinter as tk
    from tkinter import messagebox
    import gui_common

    root = gui_common.make_root("Case Merger (for the PI)", 940, 600)
    try:
        master = MasterStore.load_or_create(MASTER_FILENAME)
    except (CaseStoreError, OSError) as error:
        gui_common.show_error("Cannot open the master database", str(error))
        root.destroy()
        return 1
    MergeApp(root, master, _GuiModules(tk, messagebox))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
