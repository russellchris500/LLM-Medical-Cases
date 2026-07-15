#!/usr/bin/env python3
"""Program 1 of the LLM Medical Cases evaluation framework: the Case Editor.

Providers use this tool to create and edit medical cases and their grading
rubrics. Each provider has a preassigned provider number, and every case gets
a unique case ID of the form PPP-CCC (provider number + sequential case
number). All of a provider's cases live in a single JSON file
(provider_NNN_cases.json) that can be emailed to the principal investigator.

Requires only the Python 3 standard library.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timezone

FORMAT_VERSION = 1
FILENAME_TEMPLATE = "provider_{:03d}_cases.json"


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_case_id(provider_number, case_number):
    return "{:03d}-{:03d}".format(provider_number, case_number)


class CaseStoreError(Exception):
    """Raised when a case file is invalid or an operation is not allowed."""


class CaseStore:
    """Holds one provider's cases and reads/writes the single JSON case file."""

    def __init__(self, provider_number, path):
        if not isinstance(provider_number, int) or provider_number <= 0:
            raise CaseStoreError("Provider number must be a positive integer.")
        self.provider_number = provider_number
        self.path = path
        self.cases = []  # list of dicts, ordered by case_number
        # Highest case number ever assigned. Only increases, even when cases
        # are deleted, so a case ID is never reused for a different case.
        self._max_assigned = 0

    # ---------- persistence ----------

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise CaseStoreError("{} is not valid JSON: {}".format(path, e))

        if not isinstance(data, dict):
            raise CaseStoreError("{} does not contain a case file object.".format(path))
        if data.get("format_version") != FORMAT_VERSION:
            raise CaseStoreError(
                "{} has format_version {!r}; this program supports version {}.".format(
                    path, data.get("format_version"), FORMAT_VERSION
                )
            )
        provider_number = data.get("provider_number")
        if not isinstance(provider_number, int) or provider_number <= 0:
            raise CaseStoreError("{} has an invalid provider_number.".format(path))

        store = cls(provider_number, path)
        seen_numbers = set()
        for raw in data.get("cases", []):
            case = cls._validate_case(raw, provider_number)
            if case["case_number"] in seen_numbers:
                raise CaseStoreError(
                    "{} contains duplicate case number {}.".format(path, case["case_number"])
                )
            seen_numbers.add(case["case_number"])
            store.cases.append(case)
        store.cases.sort(key=lambda c: c["case_number"])
        highest_in_file = max(seen_numbers) if seen_numbers else 0
        stored_max = data.get("max_assigned_case_number", 0)
        if not isinstance(stored_max, int):
            stored_max = 0
        store._max_assigned = max(stored_max, highest_in_file)
        return store

    @staticmethod
    def _validate_case(raw, provider_number):
        if not isinstance(raw, dict):
            raise CaseStoreError("Case entries must be objects.")
        case_number = raw.get("case_number")
        if not isinstance(case_number, int) or case_number <= 0:
            raise CaseStoreError("Case has an invalid case_number: {!r}".format(case_number))
        expected_id = make_case_id(provider_number, case_number)
        if raw.get("case_id") != expected_id:
            raise CaseStoreError(
                "Case {} has case_id {!r}; expected {!r}.".format(
                    case_number, raw.get("case_id"), expected_id
                )
            )
        case_text = raw.get("case_text")
        if not isinstance(case_text, str) or not case_text.strip():
            raise CaseStoreError("Case {} has empty case text.".format(expected_id))
        rubric = raw.get("rubric")
        if (
            not isinstance(rubric, list)
            or not rubric
            or not all(isinstance(item, str) and item.strip() for item in rubric)
        ):
            raise CaseStoreError(
                "Case {} must have a rubric with at least one non-empty item.".format(expected_id)
            )
        return {
            "case_id": expected_id,
            "case_number": case_number,
            "case_text": case_text,
            "rubric": list(rubric),
            # Bumped every time the rubric's content changes, so grades can
            # record WHICH rubric they were made against.
            "rubric_version": (
                raw["rubric_version"]
                if isinstance(raw.get("rubric_version"), int) and raw["rubric_version"] >= 1
                else 1
            ),
            "rubric_history": (
                list(raw["rubric_history"]) if isinstance(raw.get("rubric_history"), list) else []
            ),
            "created_at": raw.get("created_at", now_iso()),
            "updated_at": raw.get("updated_at", now_iso()),
        }

    def save(self):
        data = {
            "format_version": FORMAT_VERSION,
            "provider_number": self.provider_number,
            "max_assigned_case_number": self._max_assigned,
            "cases": self.cases,
        }
        # Write atomically so an interrupted save can't corrupt the case file.
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

    # ---------- case operations ----------

    def next_case_number(self):
        # Never reuse a number, even after deletions, so case IDs stay unique
        # across the life of the study.
        return self._max_assigned + 1

    def get_case(self, case_number):
        for case in self.cases:
            if case["case_number"] == case_number:
                return case
        return None

    def add_case(self, case_text, rubric):
        case_text, rubric = self._check_content(case_text, rubric)
        number = self.next_case_number()
        self._max_assigned = number
        timestamp = now_iso()
        case = {
            "case_id": make_case_id(self.provider_number, number),
            "case_number": number,
            "case_text": case_text,
            "rubric": rubric,
            "rubric_version": 1,
            "rubric_history": [],
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        self.cases.append(case)
        return case

    def update_case(self, case_number, case_text=None, rubric=None):
        case = self.get_case(case_number)
        if case is None:
            raise CaseStoreError("No case with number {}.".format(case_number))
        new_text = case["case_text"] if case_text is None else case_text
        new_rubric = case["rubric"] if rubric is None else rubric
        new_text, new_rubric = self._check_content(new_text, new_rubric)
        if new_rubric != case["rubric"]:
            # Any change to the rubric's content is a new rubric version.
            case["rubric_version"] = int(case.get("rubric_version", 1)) + 1
        case["case_text"] = new_text
        case["rubric"] = new_rubric
        case["updated_at"] = now_iso()
        return case

    def delete_case(self, case_number):
        case = self.get_case(case_number)
        if case is None:
            raise CaseStoreError("No case with number {}.".format(case_number))
        self.cases.remove(case)
        return case

    @staticmethod
    def _check_content(case_text, rubric):
        if not isinstance(case_text, str) or not case_text.strip():
            raise CaseStoreError("Case text must not be empty.")
        if not isinstance(rubric, list) or not rubric:
            raise CaseStoreError("The rubric must contain at least one item.")
        cleaned = []
        for item in rubric:
            if not isinstance(item, str) or not item.strip():
                raise CaseStoreError("Rubric items must be non-empty text.")
            cleaned.append(item.strip())
        return case_text.rstrip(), cleaned


# ---------- window interface ----------
#
# The program is a Microsoft Windows-style application: run it (or double-
# click "Case Editor.pyw") and work in the window. Tkinter is imported
# lazily inside main() so the data logic above works everywhere.


def prompt(message):
    """Kept for backward compatibility with older helper scripts."""
    try:
        return input(message)
    except EOFError:
        print()
        raise SystemExit(0)


def find_case_files():
    return sorted(
        name for name in os.listdir(".") if re.fullmatch(r"provider_\d{3,}_cases\.json", name)
    )


def summarize(text, width=70):
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


class CaseEditorApp:
    def __init__(self, root, store, gui):
        self.root = root
        self.store = store
        self.gui = gui
        self.tk = gui.tk
        self.current_number = None  # None = editing a brand-new case
        self.loaded_snapshot = ("", "")
        self._build()
        self.refresh_list()
        if self.store.cases:
            self.select_case(self.store.cases[0]["case_number"])
        else:
            self.start_new_case()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build(self):
        tk = self.tk
        top = tk.Label(
            self.root,
            text="Provider {} - cases are saved to {} - email that file to the "
            "principal investigator".format(self.store.provider_number, self.store.path),
            anchor="w",
        )
        top.pack(fill="x", padx=8, pady=(8, 0))

        pane = tk.PanedWindow(self.root, orient="horizontal", sashrelief="raised")
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(pane)
        tk.Label(left, text="Your cases:", anchor="w").pack(fill="x")
        self.case_list = tk.Listbox(left, exportselection=False)
        self.case_list.pack(fill="both", expand=True, pady=4)
        self.case_list.bind("<<ListboxSelect>>", self.on_list_click)
        buttons = tk.Frame(left)
        buttons.pack(fill="x")
        tk.Button(buttons, text="New case", command=self.start_new_case).pack(
            side="left", padx=(0, 4)
        )
        tk.Button(buttons, text="Delete case", command=self.delete_case).pack(side="left")
        pane.add(left, minsize=240)

        right = tk.Frame(pane)
        self.header = tk.Label(right, text="", anchor="w", font=("TkDefaultFont", 10, "bold"))
        self.header.pack(fill="x")
        tk.Label(right, text="Case text (the clinical vignette / question):", anchor="w").pack(
            fill="x", pady=(6, 0)
        )
        self.case_text = self.gui.scrolledtext.ScrolledText(right, height=12, wrap="word", undo=True)
        self.case_text.pack(fill="both", expand=True, pady=4)
        tk.Label(
            right,
            text="Rubric - ONE item per line. For the answer to be correct it must "
            "match EVERY item:",
            anchor="w",
        ).pack(fill="x")
        self.rubric_text = self.gui.scrolledtext.ScrolledText(right, height=7, wrap="word", undo=True)
        self.rubric_text.pack(fill="both", expand=True, pady=4)
        bottom = tk.Frame(right)
        bottom.pack(fill="x")
        tk.Button(bottom, text="Save case", command=self.save_current).pack(side="left")
        self.status = tk.Label(bottom, text="", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=8)
        pane.add(right)

    # ---- helpers ----

    def editors_content(self):
        case_text = self.case_text.get("1.0", "end").rstrip()
        rubric = [
            line.strip()
            for line in self.rubric_text.get("1.0", "end").splitlines()
            if line.strip()
        ]
        return case_text, rubric

    def dirty(self):
        case_text, rubric = self.editors_content()
        return (case_text, "\n".join(rubric)) != self.loaded_snapshot

    def set_status(self, message):
        self.status.configure(text=message)

    def refresh_list(self, keep=None):
        self.case_list.delete(0, "end")
        for case in self.store.cases:
            self.case_list.insert(
                "end",
                "{}  [{} rubric item{}]  {}".format(
                    case["case_id"],
                    len(case["rubric"]),
                    "" if len(case["rubric"]) == 1 else "s",
                    summarize(case["case_text"], 40),
                ),
            )
        if keep is not None:
            for i, case in enumerate(self.store.cases):
                if case["case_number"] == keep:
                    self.case_list.selection_clear(0, "end")
                    self.case_list.selection_set(i)
                    self.case_list.see(i)

    def offer_save_if_dirty(self):
        """Returns False if the user cancelled the switch."""
        if not self.dirty():
            return True
        answer = self.gui.messagebox.askyesnocancel(
            "Unsaved changes",
            "This case has unsaved changes.\n\nSave them first?",
            parent=self.root,
        )
        if answer is None:
            return False
        if answer:
            return self.save_current()
        return True

    # ---- actions ----

    def on_list_click(self, _event):
        selection = self.case_list.curselection()
        if not selection:
            return
        case = self.store.cases[selection[0]]
        if case["case_number"] == self.current_number:
            return
        if not self.offer_save_if_dirty():
            self.refresh_list(keep=self.current_number)
            return
        self.select_case(case["case_number"])

    def select_case(self, case_number):
        case = self.store.get_case(case_number)
        if case is None:
            return
        self.current_number = case_number
        self.header.configure(text="Case {}".format(case["case_id"]))
        self.case_text.delete("1.0", "end")
        self.case_text.insert("1.0", case["case_text"])
        self.rubric_text.delete("1.0", "end")
        self.rubric_text.insert("1.0", "\n".join(case["rubric"]))
        self.loaded_snapshot = (case["case_text"], "\n".join(case["rubric"]))
        self.refresh_list(keep=case_number)
        self.set_status("")

    def start_new_case(self):
        if not self.offer_save_if_dirty():
            return
        self.current_number = None
        next_id = make_case_id(self.store.provider_number, self.store.next_case_number())
        self.header.configure(text="New case (will be saved as {})".format(next_id))
        self.case_text.delete("1.0", "end")
        self.rubric_text.delete("1.0", "end")
        self.loaded_snapshot = ("", "")
        self.case_list.selection_clear(0, "end")
        self.set_status("Type the case text and rubric, then click Save case.")
        self.case_text.focus_set()

    def save_current(self):
        case_text, rubric = self.editors_content()
        try:
            if self.current_number is None:
                case = self.store.add_case(case_text, rubric)
                self.current_number = case["case_number"]
            else:
                case = self.store.update_case(
                    self.current_number, case_text=case_text, rubric=rubric
                )
            self.store.save()
        except CaseStoreError as error:
            self.gui.messagebox.showerror("Cannot save", str(error), parent=self.root)
            return False
        self.loaded_snapshot = (case["case_text"], "\n".join(case["rubric"]))
        self.header.configure(text="Case {}".format(case["case_id"]))
        self.refresh_list(keep=case["case_number"])
        self.set_status("Saved case {}.".format(case["case_id"]))
        return True

    def delete_case(self):
        if self.current_number is None:
            self.gui.messagebox.showinfo(
                "Nothing to delete", "This case has not been saved yet.", parent=self.root
            )
            return
        case = self.store.get_case(self.current_number)
        if not self.gui.messagebox.askyesno(
            "Delete case",
            "Really delete case {}?\n\nThis cannot be undone.".format(case["case_id"]),
            parent=self.root,
        ):
            return
        self.store.delete_case(self.current_number)
        self.store.save()
        self.set_status("Deleted case {}.".format(case["case_id"]))
        self.current_number = None
        self.refresh_list()
        if self.store.cases:
            self.select_case(self.store.cases[0]["case_number"])
        else:
            self.start_new_case()

    def on_close(self):
        if self.offer_save_if_dirty():
            self.root.destroy()


class _GuiModules:
    """Small namespace so CaseEditorApp can reach tk pieces passed from main()."""

    def __init__(self, tk, messagebox, scrolledtext):
        self.tk = tk
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext


def open_store_with_dialogs(root):
    import gui_common

    files = find_case_files()
    if len(files) == 1:
        return CaseStore.load(files[0])
    if files:
        name = gui_common.pick_from_list(
            root,
            "Open case file",
            "More than one case file was found in this folder.\nWhich one do you "
            "want to open?",
            files,
        )
        return CaseStore.load(name) if name else None
    number = gui_common.ask_int(
        root, "Provider number", "Enter your preassigned provider number:"
    )
    if not number:
        return None
    path = FILENAME_TEMPLATE.format(number)
    if os.path.exists(path):
        return CaseStore.load(path)
    store = CaseStore(number, path)
    store.save()
    return store


def main():
    import tkinter as tk
    from tkinter import messagebox, scrolledtext
    import gui_common

    root = gui_common.make_root("Case Editor (for providers)", 980, 640)
    root.withdraw()
    try:
        store = open_store_with_dialogs(root)
    except (CaseStoreError, OSError) as error:
        gui_common.show_error("Cannot open the case file", str(error))
        root.destroy()
        return 1
    if store is None:
        root.destroy()
        return 0
    CaseEditorApp(root, store, _GuiModules(tk, messagebox, scrolledtext))
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
