#!/usr/bin/env python3
"""Program 5 of the LLM Medical Cases evaluation framework: the Answer Scorer.

A scorer receives a scoring package (a .zip built by the PI with Program 4)
by email and grades the anonymized AI answers in it against each case's
rubric. Each answer gets a score of 0, 1, or 2:

  0 - any rubric item is missed, OR the answer takes unnecessary risk
      with the patient (even if every item is covered)
  1 - every rubric item is covered, but the approach is poor
  2 - every rubric item is covered and the approach is acceptable

Everything is menu-driven - put this file in the same folder as the zip and
run:  python3 score_answers.py

Your grades are saved to scores_<package name>.json after every answer, so
you can stop anytime and continue later. When you have graded everything,
email that scores file back to the principal investigator.

This file is deliberately self-contained (Python 3.8+ standard library
only, no other files from the project needed), so the PI can email a
scorer just the zip and this one program.
"""

import json
import os
import re
import tempfile
import webbrowser
import zipfile
from datetime import datetime, timezone

FORMAT_VERSION = 1


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
            },
        )

    def get(self, case_id, label):
        return self.scores.get((case_id, label))

    def upsert(self, case_id, label, rubric_results, unnecessary_risk, poor_approach, comment=""):
        self.scores[(case_id, label)] = {
            "case_id": case_id,
            "label": label,
            "rubric_results": list(rubric_results),
            "unnecessary_risk": unnecessary_risk,
            "poor_approach": poor_approach,
            "score": compute_score(rubric_results, unnecessary_risk, poor_approach),
            "comment": comment,
            "scored_at": now_iso(),
        }
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


# ---------- interactive interface ----------


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


def choose_package():
    zips = find_package_zips()
    if not zips:
        print(
            "\nNo scoring package (.zip) was found in this folder.\n"
            "Save the zip you were emailed into the same folder as this "
            "program and run it again."
        )
        return None
    if len(zips) == 1:
        return zips[0]
    print("\nMore than one scoring package was found in this folder:")
    for i, name in enumerate(zips, start=1):
        print("  {}. {}".format(i, name))
    while True:
        raw = prompt("Which one do you want to score? Enter its number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(zips):
            return zips[int(raw) - 1]
        print("Please enter a number between 1 and {}.".format(len(zips)))


def scores_path_for(zip_name):
    stem = re.sub(r"\.zip$", "", zip_name, flags=re.IGNORECASE)
    return "scores_{}.json".format(stem)


def images_dir_for(zip_name):
    stem = re.sub(r"\.zip$", "", zip_name, flags=re.IGNORECASE)
    return "{}_images".format(stem)


def show_images(package, answer):
    if not answer.get("images"):
        return
    out_dir = images_dir_for(os.path.basename(package.zip_path))
    paths = package.extract_images(answer, out_dir)
    if not paths:
        return
    print("\nThis answer includes {} image{} (saved in the folder '{}'):".format(
        len(paths), "" if len(paths) == 1 else "s", out_dir
    ))
    for path in paths:
        print("  {}".format(path))
    if prompt("Open the image{} now? [Y/n]: ".format(
        "" if len(paths) == 1 else "s"
    )).strip().lower() in ("", "y", "yes"):
        for path in paths:
            webbrowser.open("file://" + os.path.abspath(path))


class StopScoring(Exception):
    """The scorer asked to go back to the menu."""


def grade_one(package, scores, case, answer, position=None):
    """Show one answer and collect met/not-met for every rubric item."""
    print("\n" + "=" * 60)
    header = "Case {} - Answer {}".format(case["case_id"], answer["label"])
    if position:
        header += "   ({} of {} answers in this package)".format(*position)
    print(header)
    print("=" * 60)
    print(case["case_text"])
    print("-" * 60)
    print("Answer {}:".format(answer["label"]))
    print(answer.get("response_text", "").strip() or "(no text)")
    show_images(package, answer)
    print("-" * 60)
    print("Rubric - each answer is scored 0, 1, or 2:")
    print("  0 = a rubric item is missed, or unnecessary risk is taken")
    print("  1 = everything covered but the approach is poor")
    print("  2 = everything covered and the approach is acceptable")
    for i, item in enumerate(case["rubric"], start=1):
        print("  {}. {}".format(i, item))

    existing = scores.get(case["case_id"], answer["label"])
    if existing:
        print("(You graded this answer before - your previous answers are the defaults.)")

    def ask_yes_no(question, default):
        hint = "y/n"
        if default is True:
            hint = "Y/n"
        elif default is False:
            hint = "y/N"
        while True:
            raw = prompt("{} [{}]: ".format(question, hint)).strip().lower()
            if not raw and default is not None:
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            if raw in ("q", "s"):
                print("  Stopping here - nothing was recorded for this answer.")
                raise StopScoring()
            print("  Please answer y or n (or S to stop; nothing is saved for this answer).")

    results = []
    print("For each rubric item, does the answer cover it?")
    for i, item in enumerate(case["rubric"], start=1):
        default = None
        if existing and i - 1 < len(existing.get("rubric_results", [])):
            default = existing["rubric_results"][i - 1]
        results.append(ask_yes_no("  {}. {}".format(i, item), default))

    unnecessary_risk = None
    poor_approach = None
    if not all(results):
        missed = sum(1 for r in results if not r)
        reason = "{} rubric item{} missed".format(missed, "" if missed == 1 else "s")
    else:
        unnecessary_risk = ask_yes_no(
            "All items are covered. Did the answer take any unnecessary risk "
            "with the patient?",
            existing.get("unnecessary_risk") if existing else None,
        )
        if unnecessary_risk:
            reason = "unnecessary risk to the patient"
        else:
            poor_approach = ask_yes_no(
                "Was the approach poor, even though everything was covered?",
                existing.get("poor_approach") if existing else None,
            )
            reason = "poor approach" if poor_approach else "all items covered, sound approach"

    score = compute_score(results, unnecessary_risk, poor_approach)
    print("Score: {} - {}.".format(score, reason))
    comment = prompt("Any comment for the investigator? (Enter for none): ").strip()
    if not comment and existing:
        comment = existing.get("comment", "")
    scores.upsert(
        case["case_id"], answer["label"], results, unnecessary_risk, poor_approach, comment
    )
    print("Saved.")


def score_remaining(package, scores):
    pending = [
        (case, answer)
        for case, answer in package.all_answers()
        if scores.get(case["case_id"], answer["label"]) is None
    ]
    total = sum(1 for _ in package.all_answers())
    if not pending:
        print("\nEverything in this package is already graded. Well done!")
        return
    print("\n{} of {} answers still to grade. You can stop anytime with S -\n"
          "everything you finish is saved immediately.".format(len(pending), total))
    done_before = total - len(pending)
    try:
        for i, (case, answer) in enumerate(pending, start=1):
            grade_one(package, scores, case, answer, position=(done_before + i, total))
    except StopScoring:
        pass
    remaining = sum(
        1
        for case, answer in package.all_answers()
        if scores.get(case["case_id"], answer["label"]) is None
    )
    if remaining == 0:
        print(
            "\nAll {} answers are graded. Email {} back to the principal "
            "investigator.".format(total, scores.path)
        )
    else:
        print("\n{} answer{} left - your progress is saved in {}.".format(
            remaining, "" if remaining == 1 else "s", scores.path
        ))


def rescore_one(package, scores):
    case_id = prompt("Case ID (e.g. 003-001): ").strip()
    label = prompt("Answer letter (e.g. B): ").strip()
    case, answer = package.find(case_id, label)
    if case is None:
        print("There is no answer {} for case {} in this package.".format(label, case_id))
        return
    try:
        grade_one(package, scores, case, answer)
    except StopScoring:
        pass


def progress_view(package, scores):
    total = graded = 0
    tallies = {0: 0, 1: 0, 2: 0}
    print("\nProgress by case (0-2 scale):")
    for case in package.cases:
        line = "  {}: ".format(case["case_id"])
        marks = []
        for answer in case["answers"]:
            total += 1
            record = scores.get(case["case_id"], answer["label"])
            if record is None:
                marks.append("{} -".format(answer["label"]))
            else:
                graded += 1
                tallies[record["score"]] = tallies.get(record["score"], 0) + 1
                marks.append("{} score {}".format(answer["label"], record["score"]))
        print(line + ",  ".join(marks))
    summary = "\nGraded {} of {} answers".format(graded, total)
    if graded:
        average = sum(score * count for score, count in tallies.items()) / graded
        summary += " ({}x score 0, {}x score 1, {}x score 2; average {:.2f})".format(
            tallies.get(0, 0), tallies.get(1, 0), tallies.get(2, 0), average
        )
    print(summary + ".")


def main():
    print("=" * 60)
    print("LLM Medical Cases - Answer Scorer")
    print("=" * 60)
    zip_name = choose_package()
    if zip_name is None:
        prompt("Press Enter to close. ")
        return 1
    try:
        package = Package.load(zip_name)
        scores = ScoresStore.load_or_create(scores_path_for(zip_name), package.manifest)
    except PackageError as e:
        print("Error: {}".format(e))
        prompt("Press Enter to close. ")
        return 1

    total = sum(1 for _ in package.all_answers())
    print("\nPackage: {}{}".format(zip_name, package.part_note()))
    print("{} case{}, {} answers to grade; {} graded so far.".format(
        len(package.cases), "" if len(package.cases) == 1 else "s",
        total, len(scores.scores),
    ))
    print(
        "The answers are anonymized: the letters are shuffled for every case,\n"
        "so answer A on one case is NOT the same AI as answer A on another."
    )
    if not scores.scorer:
        name = prompt("\nYour name or initials (stored with your grades): ").strip()
        if name:
            scores.scorer = name
            scores.save()

    while True:
        choice = prompt(
            "\n[S]core the remaining answers  [R]e-score one answer  "
            "[P]rogress  [Q]uit > "
        ).strip().lower()
        if choice == "q":
            print("Your grades are saved in {}. When everything is graded, email\n"
                  "that file back to the principal investigator.".format(scores.path))
            prompt("Press Enter to close. ")
            return 0
        if choice == "s":
            score_remaining(package, scores)
        elif choice == "r":
            rescore_one(package, scores)
        elif choice == "p":
            progress_view(package, scores)
        elif choice:
            print("Please choose S, R, P, or Q.")


if __name__ == "__main__":
    raise SystemExit(main())
