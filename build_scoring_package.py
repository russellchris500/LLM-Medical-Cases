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

Everything is menu-driven - just run:  python3 build_scoring_package.py

Requires only the Python 3 standard library.
"""

import json
import os
import random
import re
import string
import tempfile
import zipfile

from case_editor import CaseStoreError, now_iso, prompt
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
                "model_id": record["model_id"],
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
        "key": key,
    }
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    for warning in self_identification_warnings(manifest["cases"]):
        warn("Blinding note: " + warning)
    return zip_path, key_path


# ---------- interactive flow ----------


def choose_package_cases(master, answers, case_sets):
    """Pick cases from those that actually have usable answers."""
    answered = {
        case_id: master.cases[case_id]
        for case_id in answers.answered_case_ids()
        if case_id in master.cases
    }
    orphans = [c for c in answers.answered_case_ids() if c not in master.cases]
    if orphans:
        print(
            "Note: {} answered case{} no longer in master_cases.json and cannot be\n"
            "packaged (the case text lives only there): {}\n"
            "Re-merge the provider's file to bring {} back.".format(
                len(orphans), " is" if len(orphans) == 1 else "s are",
                ", ".join(orphans), "it" if len(orphans) == 1 else "them",
            )
        )
    if not answered:
        print("No cases have usable answers yet - run run_llms.py first.")
        return None

    per_case = {c: len([1 for (cid, m), r in answers.answers.items()
                        if cid == c and r.get("status") in OK_STATUSES])
                for c in answered}
    while True:
        print("\nChoose the cases for the scorer ({} cases have answers).".format(len(answered)))
        print("  [A]ll of them")
        print("  [E]nter case IDs or ranges")
        print("  [L]ist cases with their answer counts")
        print("  [S]aved case set                     ({} saved)".format(len(case_sets.sets)))
        choice = prompt("Choose A, E, L, or S (blank to cancel): ").strip().lower()
        if not choice:
            return None
        if choice == "a":
            return sort_case_ids(answered)
        if choice == "l":
            for case_id in sort_case_ids(answered):
                print("  {}  {} answer{}".format(
                    case_id, per_case[case_id], "" if per_case[case_id] == 1 else "s"
                ))
            continue
        if choice == "s":
            names = sorted(case_sets.sets, key=str.lower)
            if not names:
                print("There are no saved case sets yet.")
                continue
            for i, set_name in enumerate(names, start=1):
                print("  {}. {}  ({} cases)".format(
                    i, set_name, len(case_sets.sets[set_name]["case_ids"])
                ))
            raw = prompt("Which set? Enter its number (blank to cancel): ").strip()
            if not raw.isdigit() or not 1 <= int(raw) <= len(names):
                continue
            present, missing = case_sets.resolve(names[int(raw) - 1], answered)
            if missing:
                print("  {} case{} in the set have no usable answers and will be "
                      "left out: {}".format(
                          len(missing), "" if len(missing) == 1 else "s", ", ".join(missing)))
            if present:
                return present
            print("None of that set's cases have answers.")
            continue
        if choice == "e":
            expression = prompt("Enter cases (e.g. 003-001..003-020): ").strip()
            if not expression:
                continue
            try:
                selected, warnings = parse_selection(expression, answered)
            except SelectionError as e:
                print(str(e))
                continue
            for warning in warnings:
                print("  Note: {}".format(warning))
            if selected:
                return selected
            print("That selection matched no cases with answers.")
            continue
        print("Please choose A, E, L, or S.")


def choose_package_models(answers, case_ids):
    available = answers.model_ids()
    usable = [
        m for m in available
        if any(
            (answers.get(c, m) or {}).get("status") in OK_STATUSES for c in case_ids
        )
    ]
    if not usable:
        print("None of the models have usable answers for those cases.")
        return None
    display = {}
    for model_id in usable:
        for c in case_ids:
            record = answers.get(c, model_id)
            if record is not None:
                display[model_id] = record.get("model_display_name") or model_id
                break
    print("\nChoose the AIs whose answers go to the scorer:")
    for i, model_id in enumerate(usable, start=1):
        count = sum(
            1 for c in case_ids
            if (answers.get(c, model_id) or {}).get("status") in OK_STATUSES
        )
        print("  {}. {:22s} answers for {}/{} of the chosen cases".format(
            i, display.get(model_id, model_id), count, len(case_ids)
        ))
    raw = prompt("Enter numbers (e.g. 1,3) or A for all (blank to cancel): ").strip()
    if not raw:
        return None
    if raw.lower() in ("a", "all"):
        return usable
    chosen = []
    for part in raw.replace(",", " ").split():
        if not part.isdigit() or not 1 <= int(part) <= len(usable):
            print("There is no model number {}.".format(part))
            return None
        if usable[int(part) - 1] not in chosen:
            chosen.append(usable[int(part) - 1])
    return chosen or None


def coverage_check(master, answers, case_ids, model_ids):
    """Handle holes in the case x model grid; returns the final case list."""
    holes = []
    failed = []
    changed = []
    for case_id in case_ids:
        for model_id in model_ids:
            record = answers.get(case_id, model_id)
            if record is None:
                holes.append((case_id, model_id))
            elif record.get("status") not in OK_STATUSES:
                failed.append((case_id, model_id))
            elif record.get("case_sha256") != case_hash(master.cases[case_id]):
                changed.append((case_id, model_id))
    if failed:
        print("\n{} answer{} failed when collected and will not be included:".format(
            len(failed), "" if len(failed) == 1 else "s"
        ))
        for case_id, model_id in failed[:10]:
            print("  {} x {}".format(case_id, model_id))
    if changed:
        print("\nNote: for {} answer{} the case wording changed after the answer was\n"
              "collected (shown to the scorer as-is):".format(
                  len(changed), "" if len(changed) == 1 else "s"))
        for case_id, model_id in changed[:10]:
            print("  {} x {}".format(case_id, model_id))
    incomplete = sorted({c for c, _ in holes + failed}, key=split_case_id)
    if incomplete:
        print("\n{} case{} not have an answer from every chosen AI:".format(
            len(incomplete), " does" if len(incomplete) == 1 else "s do"
        ))
        for case_id in incomplete[:10]:
            have = [m for m in model_ids
                    if (answers.get(case_id, m) or {}).get("status") in OK_STATUSES]
            print("  {}  (has: {})".format(case_id, ", ".join(have) or "none"))
        while True:
            choice = prompt(
                "[I]nclude those cases with the answers they have  "
                "[E]xclude incomplete cases  [C]ancel: "
            ).strip().lower()
            if choice == "c":
                return None
            if choice == "i":
                break
            if choice == "e":
                case_ids = [c for c in case_ids if c not in set(incomplete)]
                break
    final = [
        c for c in case_ids
        if any((answers.get(c, m) or {}).get("status") in OK_STATUSES for m in model_ids)
    ]
    if not final:
        print("Nothing left to package.")
        return None
    return final


def maybe_split(zip_path, name, case_ids, model_ids, master, answers, key_path):
    size = os.path.getsize(zip_path)
    if size <= EMAIL_SIZE_LIMIT:
        return [zip_path]
    print(
        "\nThe package is {:.1f} MB, which may be too large to email "
        "(most mail systems cap attachments around 25 MB).".format(size / (1024 * 1024))
    )
    choice = prompt(
        "[S]plit it into parts of at most 20 MB  [K]eep it as one file: "
    ).strip().lower()
    if choice != "s":
        return [zip_path]

    # Split by whole cases: pack greedily until a part would exceed the cap.
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
    # The split parts have their own keys; keep only those and remove the
    # oversized original so there's one obvious thing to send per part.
    os.unlink(zip_path)
    os.unlink(key_path)
    print("Split into {} parts (each with its own key file):".format(len(parts)))
    return parts


def build_flow(master, answers, case_sets):
    case_ids = choose_package_cases(master, answers, case_sets)
    if not case_ids:
        return
    model_ids = choose_package_models(answers, case_ids)
    if not model_ids:
        return
    case_ids = coverage_check(master, answers, case_ids, model_ids)
    if not case_ids:
        return

    while True:
        name = prompt("\nName for this package (e.g. pilot20-drsmith): ").strip()
        if not name:
            return
        if not SET_NAME_RE.match(name):
            print(
                "Package names may only use letters, digits, dots, dashes, and "
                "underscores (up to 40 characters)."
            )
            continue
        zip_path = os.path.join(PACKAGES_DIR, name + ".zip")
        if os.path.exists(zip_path):
            if prompt(
                "A package with that name exists. Type yes to replace it: "
            ).strip().lower() != "yes":
                continue
        break

    zip_path, key_path = build_package(name, case_ids, model_ids, master, answers)
    parts = maybe_split(zip_path, name, case_ids, model_ids, master, answers, key_path)

    print("\nBuilt:")
    for part in parts:
        print("  {}  ({:.1f} MB) - email this to the scorer.".format(
            part, os.path.getsize(part) / (1024 * 1024)
        ))
    print(
        "\nThe matching *_KEY_DO_NOT_SEND.json file(s) in {} stay with you -\n"
        "they reveal which AI wrote each answer. NEVER send them to a scorer.".format(
            PACKAGES_DIR
        )
    )


def list_packages():
    if not os.path.isdir(PACKAGES_DIR):
        print("\nNo packages have been built yet.\n")
        return
    zips = sorted(f for f in os.listdir(PACKAGES_DIR) if f.endswith(".zip"))
    if not zips:
        print("\nNo packages have been built yet.\n")
        return
    print("\nBuilt packages in {}:".format(PACKAGES_DIR))
    for filename in zips:
        path = os.path.join(PACKAGES_DIR, filename)
        note = ""
        try:
            with zipfile.ZipFile(path) as bundle:
                manifest = json.loads(bundle.read("package.json"))
            note = "{} cases, built {}".format(
                manifest.get("num_cases"), manifest.get("created_at", "")[:10]
            )
        except Exception:
            note = "unreadable"
        print("  {}  ({:.1f} MB; {})".format(
            filename, os.path.getsize(path) / (1024 * 1024), note
        ))
    print()


def main():
    print("=" * 60)
    print("LLM Medical Cases - Scoring Package Builder (for the PI)")
    print("=" * 60)
    try:
        answers = AnswersStore.load_or_create()
        case_sets = CaseSetStore.load_or_create()
        master = MasterStore.load(MASTER_FILENAME) if os.path.exists(MASTER_FILENAME) else None
    except (CaseStoreError, OSError) as e:
        print("Error: {}".format(e))
        prompt("Press Enter to close. ")
        return 1
    if master is None:
        print("\nNo master case database ({}) was found in this folder.".format(MASTER_FILENAME))
        prompt("Press Enter to close. ")
        return 1
    usable = answers.answered_case_ids()
    print("Answers available: {} covering {} case{} and {} model{}.".format(
        sum(1 for r in answers.answers.values() if r.get("status") in OK_STATUSES),
        len(usable), "" if len(usable) == 1 else "s",
        len(answers.model_ids()), "" if len(answers.model_ids()) == 1 else "s",
    ))

    while True:
        choice = prompt(
            "\n[B]uild a package  [L]ist built packages  [Q]uit > "
        ).strip().lower()
        try:
            if choice == "q":
                prompt("Press Enter to close. ")
                return 0
            if choice == "b":
                build_flow(master, answers, case_sets)
            elif choice == "l":
                list_packages()
            elif choice:
                print("Please choose B, L, or Q.")
        except CaseStoreError as e:
            print("Error: {}".format(e))


if __name__ == "__main__":
    raise SystemExit(main())
