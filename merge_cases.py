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

from case_editor import FORMAT_VERSION, CaseStore, CaseStoreError, prompt

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


# ---------- interactive interface ----------


def preview(case, limit=68):
    flat = " ".join(case["case_text"].split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def ask_yes_no(question, default=False):
    answer = prompt("{} [{}]: ".format(question, "Y/n" if default else "y/N")).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def on_conflict(existing, incoming):
    old = parse_timestamp(existing.get("updated_at"))
    new = parse_timestamp(incoming.get("updated_at"))
    incoming_newer = old is not None and new is not None and new > old
    print("\nCase {} differs between the master and the incoming file:".format(existing["case_id"]))
    print(
        "  in master : edited {}  ({} rubric items)  {}".format(
            existing.get("updated_at"), len(existing["rubric"]), preview(existing)
        )
    )
    print(
        "  incoming  : edited {}  ({} rubric items)  {}".format(
            incoming.get("updated_at"), len(incoming["rubric"]), preview(incoming)
        )
    )
    if incoming_newer:
        print("  The incoming version is newer.")
    return ask_yes_no(
        "  Replace the master version with the incoming one?", default=incoming_newer
    )


def on_missing(existing):
    print(
        "\nCase {} is in the master but not in the incoming file "
        "(the provider may have deleted it).".format(existing["case_id"])
    )
    print("  {}".format(preview(existing)))
    return ask_yes_no("  Remove it from the master too?", default=False)


def print_report(source, provider_number, report):
    print("\nResult for {} (provider {}):".format(source, provider_number))
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
            print("  {:3d} {}: {}".format(len(report[key]), label, ", ".join(report[key])))
    if not any(report.values()):
        print("  nothing to do (the incoming file has no cases)")


def list_master(master):
    if not master.cases:
        print("\nThe master file has no cases yet.\n")
        return
    by_provider = {}
    for case in master.cases.values():
        by_provider.setdefault(case["provider_number"], []).append(case)
    print("\nMaster file {}: {} cases from {} provider(s)".format(
        master.path, len(master.cases), len(by_provider)
    ))
    for provider_number in sorted(by_provider):
        cases = sorted(by_provider[provider_number], key=lambda c: c["case_number"])
        print("\nProvider {} ({} case{}):".format(
            provider_number, len(cases), "" if len(cases) == 1 else "s"
        ))
        for case in cases:
            print("  {}  [{} rubric item{}]  {}".format(
                case["case_id"],
                len(case["rubric"]),
                "" if len(case["rubric"]) == 1 else "s",
                preview(case),
            ))
    print()


def find_provider_files():
    return sorted(
        name for name in os.listdir(".") if re.fullmatch(r"provider_\d{3,}_cases\.json", name)
    )


def choose_provider_files():
    """Show the provider files in this folder and ask which to merge."""
    files = find_provider_files()
    if not files:
        print("\nNo provider case files (provider_NNN_cases.json) were found in this folder.")
        print("Copy the files the providers emailed you into this folder, then try again.")
        name = prompt("Or type a file name to merge (press Enter to go back): ").strip()
        return [name] if name else []

    print("\nProvider case files found in this folder:")
    for i, name in enumerate(files, start=1):
        print("  {}. {}".format(i, name))
    raw = prompt(
        "Which do you want to merge? Enter numbers like 1,3 or A for all\n"
        "(or type a different file name; press Enter to go back): "
    ).strip()
    if not raw:
        return []
    if raw.lower() in ("a", "all"):
        return files
    if re.fullmatch(r"[\d,\s]+", raw):
        chosen = []
        for part in raw.replace(",", " ").split():
            number = int(part)
            if not 1 <= number <= len(files):
                print("There is no file number {}. Please try again.".format(number))
                return []
            if files[number - 1] not in chosen:
                chosen.append(files[number - 1])
        return chosen
    return [raw]


def merge_files(master):
    paths = choose_provider_files()
    if not paths:
        return
    # Validate every chosen file up front so one bad file stops the merge
    # before the master is touched.
    provider_stores = []
    for path in paths:
        try:
            provider_stores.append(CaseStore.load(path))
        except (CaseStoreError, OSError) as e:
            print("\nProblem with {}: {}".format(path, e))
            print("Nothing was merged. Ask the provider to re-send the file, then try again.\n")
            return

    for path, provider_store in zip(paths, provider_stores):
        report = master.merge_provider(
            provider_store, on_conflict=on_conflict, on_missing=on_missing
        )
        print_report(path, provider_store.provider_number, report)
    master.save()
    print("\nSaved {} ({} cases total).\n".format(master.path, len(master.cases)))


def main():
    print("=" * 60)
    print("LLM Medical Cases - Case Merger (for the PI)")
    print("=" * 60)
    try:
        master = MasterStore.load_or_create(MASTER_FILENAME)
    except (CaseStoreError, OSError) as e:
        print("Error: {}".format(e))
        prompt("Press Enter to close. ")
        return 1
    if os.path.exists(MASTER_FILENAME):
        print("Master file {}: {} case{} loaded.\n".format(
            MASTER_FILENAME, len(master.cases), "" if len(master.cases) == 1 else "s"
        ))
    else:
        print("No master file yet - {} will be created on the first merge.\n".format(
            MASTER_FILENAME
        ))

    while True:
        choice = prompt("[M]erge provider files  [L]ist the master  [Q]uit > ").strip().lower()
        if choice == "q":
            print("The master database is saved in {}.".format(MASTER_FILENAME))
            prompt("Press Enter to close. ")
            return 0
        if choice == "m":
            merge_files(master)
        elif choice == "l":
            list_master(master)
        elif choice:
            print("Please choose M, L, or Q.")


if __name__ == "__main__":
    raise SystemExit(main())
