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


# ---------- interactive interface ----------


def prompt(message):
    try:
        return input(message)
    except EOFError:
        print()
        raise SystemExit(0)


def read_multiline(header):
    print(header)
    print("(Type the text; blank lines are allowed. Finish with a single '.' on its own line.)")
    lines = []
    while True:
        line = prompt("> ")
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).rstrip()


def read_rubric_items(existing_count=0):
    print("Enter rubric items, one per line. Press Enter on an empty line to finish.")
    print("(For an answer to be correct it must match EVERY rubric item.)")
    items = []
    while True:
        item = prompt("  rubric item {}: ".format(existing_count + len(items) + 1)).strip()
        if not item:
            break
        items.append(item)
    return items


def choose_case(store, action):
    if not store.cases:
        print("There are no cases yet.\n")
        return None
    list_cases(store)
    raw = prompt("Case number to {} (blank to cancel): ".format(action)).strip()
    if not raw:
        return None
    match = re.fullmatch(r"(?:\d{3}-)?0*(\d+)", raw)
    if not match:
        print("Please enter a case number such as 2 or 003-002.\n")
        return None
    case = store.get_case(int(match.group(1)))
    if case is None:
        print("No case with that number.\n")
        return None
    return case


def summarize(text, width=70):
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def list_cases(store):
    print("\nCases for provider {} ({} total):".format(store.provider_number, len(store.cases)))
    for case in store.cases:
        print(
            "  {}  [{} rubric item{}]  {}".format(
                case["case_id"],
                len(case["rubric"]),
                "" if len(case["rubric"]) == 1 else "s",
                summarize(case["case_text"]),
            )
        )
    print()


def show_case(case):
    print("\nCase {}".format(case["case_id"]))
    print("Created: {}   Last edited: {}".format(case["created_at"], case["updated_at"]))
    print("-" * 60)
    print(case["case_text"])
    print("-" * 60)
    print("Rubric (the answer must match EVERY item to be correct):")
    for i, item in enumerate(case["rubric"], start=1):
        print("  {}. {}".format(i, item))
    print()


def create_case(store):
    print("\n--- New case (will be {} ) ---".format(make_case_id(store.provider_number, store.next_case_number())))
    case_text = read_multiline("Enter the case text:")
    if not case_text.strip():
        print("Cancelled: the case text was empty.\n")
        return
    rubric = read_rubric_items()
    if not rubric:
        print("Cancelled: a case needs at least one rubric item.\n")
        return
    case = store.add_case(case_text, rubric)
    store.save()
    print("Saved case {}.\n".format(case["case_id"]))


def edit_rubric(store, case):
    while True:
        print("Rubric for case {}:".format(case["case_id"]))
        for i, item in enumerate(case["rubric"], start=1):
            print("  {}. {}".format(i, item))
        choice = prompt("Rubric: [A]dd, [E]dit #, [R]emove #, or Enter when done: ").strip().lower()
        if not choice:
            return
        rubric = list(case["rubric"])
        if choice == "a":
            rubric.extend(read_rubric_items(existing_count=len(rubric)))
        elif choice[0] in ("e", "r"):
            number = choice[1:].strip() or prompt("Which item number? ").strip()
            if not number.isdigit() or not 1 <= int(number) <= len(rubric):
                print("There is no rubric item {}.".format(number or "?"))
                continue
            index = int(number) - 1
            if choice[0] == "r":
                if len(rubric) == 1:
                    print("A case must keep at least one rubric item.")
                    continue
                removed = rubric.pop(index)
                print("Removed: {}".format(removed))
            else:
                print("Current text: {}".format(rubric[index]))
                new_item = prompt("New text (blank to keep): ").strip()
                if new_item:
                    rubric[index] = new_item
        else:
            print("Please choose A, E, or R (e.g. 'e2' edits item 2).")
            continue
        store.update_case(case["case_number"], rubric=rubric)
        store.save()
        print("Rubric saved.")


def edit_case(store):
    case = choose_case(store, "edit")
    if case is None:
        return
    show_case(case)
    if prompt("Replace the case text? [y/N]: ").strip().lower() == "y":
        new_text = read_multiline("Enter the new case text:")
        if new_text.strip():
            store.update_case(case["case_number"], case_text=new_text)
            store.save()
            print("Case text saved.")
        else:
            print("Kept the existing case text (new text was empty).")
    edit_rubric(store, case)
    print("Finished editing case {}.\n".format(case["case_id"]))


def view_case(store):
    case = choose_case(store, "view")
    if case is not None:
        show_case(case)


def delete_case(store):
    case = choose_case(store, "delete")
    if case is None:
        return
    show_case(case)
    if prompt("Really delete case {}? Type 'yes' to confirm: ".format(case["case_id"])).strip().lower() == "yes":
        store.delete_case(case["case_number"])
        store.save()
        print("Deleted case {}.\n".format(case["case_id"]))
    else:
        print("Not deleted.\n")


def ask_provider_number():
    while True:
        raw = prompt("Enter your preassigned provider number: ").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("The provider number must be a positive whole number.")


def choose_from_list(names, question):
    """Ask the user to pick one entry from a numbered list of file names."""
    for i, name in enumerate(names, start=1):
        print("  {}. {}".format(i, name))
    while True:
        raw = prompt(question).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        print("Please enter a number between 1 and {}.".format(len(names)))


def open_store():
    """Find this provider's case file in the current folder (or create one)."""
    existing = sorted(
        name for name in os.listdir(".") if re.fullmatch(r"provider_\d{3,}_cases\.json", name)
    )
    if len(existing) == 1:
        return CaseStore.load(existing[0])
    if len(existing) > 1:
        print("More than one case file was found in this folder:")
        return CaseStore.load(choose_from_list(existing, "Which one do you want to open? Enter its number: "))

    provider_number = ask_provider_number()
    path = FILENAME_TEMPLATE.format(provider_number)
    if os.path.exists(path):
        return CaseStore.load(path)
    store = CaseStore(provider_number, path)
    store.save()
    print("Created new case file {}.".format(path))
    return store


def main():
    print("=" * 60)
    print("LLM Medical Cases - Case Editor")
    print("=" * 60)
    try:
        store = open_store()
    except (CaseStoreError, OSError) as e:
        print("Error: {}".format(e))
        prompt("Press Enter to close. ")
        return 1

    print(
        "Provider {} - {} case{} in {}\n".format(
            store.provider_number,
            len(store.cases),
            "" if len(store.cases) == 1 else "s",
            store.path,
        )
    )

    actions = {
        "n": create_case,
        "l": lambda s: list_cases(s),
        "v": view_case,
        "e": edit_case,
        "d": delete_case,
    }
    while True:
        choice = prompt(
            "[N]ew case  [L]ist  [V]iew  [E]dit  [D]elete  [Q]uit > "
        ).strip().lower()
        if choice == "q":
            print(
                "All changes are saved in {}. Email that file to the principal investigator.".format(
                    store.path
                )
            )
            prompt("Press Enter to close. ")
            return 0
        action = actions.get(choice)
        if action:
            try:
                action(store)
            except CaseStoreError as e:
                print("Error: {}\n".format(e))
        elif choice:
            print("Please choose N, L, V, E, D, or Q.")


if __name__ == "__main__":
    raise SystemExit(main())
