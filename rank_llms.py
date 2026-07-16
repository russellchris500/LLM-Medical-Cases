#!/usr/bin/env python3
"""Program 6 of the LLM Medical Cases evaluation framework: the LLM Ranker.

The principal investigator (PI) runs this after scorers email back their
scores_<package>.json files. It joins those grades to the PI's key files
(scoring_packages/*_KEY_DO_NOT_SEND.json, matched by package_id), then
rates every LLM AND every case on an Elo scale.

How the rating works, in plain language:
- Every graded answer is treated as one match between an LLM and a case.
  A score of 2 is a win for the LLM, 1 is a draw, and 0 is a loss (the
  case "beat" the LLM).
- Ratings are NOT updated game-by-game the way chess Elo is - that would
  make the result depend on the arbitrary order the games are processed.
  Instead, all LLM and case ratings are fitted at once by logistic
  regression (maximum likelihood on the standard Elo win-probability
  curve), which uses every match simultaneously and gives one stable,
  order-independent answer.
- The fitted ratings sit on the familiar Elo scale: the average case is
  anchored at 1500, and the gap between an LLM and a case gives the
  predicted chance the LLM handles that case well
  (P = 1 / (1 + 10^((case - LLM) / 400))).
- So a higher LLM rating means a stronger model, and a higher case rating
  means a harder case.
- To keep ratings finite when a model wins or loses everything, every LLM
  and every case is given one imaginary drawn match against an average
  (1500-rated) opponent.
- When the SAME answer was graded by several scorers, every grade counts
  as its own match: agreement strengthens the rating, disagreement
  averages out. Duplicate grades from the SAME scorer (a stray copy or an
  outdated scores file) are ignored - only their most recently saved
  grade counts - and the window reports how often multiple scorers
  agreed (inter-rater agreement).

It is a window-based program - run it (or double-click "Rank LLMs.pyw");
the ranking appears as soon as the files are read.
Requires only the Python 3 standard library.
"""

import csv
import json
import math
import os
import re

from eval_common import split_case_id

ELO_CENTER = 1500.0
ELO_SCALE = math.log(10) / 400.0  # converts Elo differences to logistic units
PACKAGES_DIR = "scoring_packages"
RESULTS_DIR = "ranking_results"
KEY_FILE_RE = re.compile(r".*_KEY_DO_NOT_SEND\.json$", re.IGNORECASE)
SCORES_FILE_RE = re.compile(r"scores_.*\.json$", re.IGNORECASE)

# score (0/1/2) -> match result for the LLM (loss / draw / win)
RESULT_FOR_SCORE = {0: 0.0, 1: 0.5, 2: 1.0}


def expected_win(r_llm, r_case):
    """Elo win probability of the LLM against the case."""
    return 1.0 / (1.0 + 10 ** ((r_case - r_llm) / 400.0))


def fit_ratings(matches, prior_weight=1.0, tol=0.001, max_iter=500):
    """Fit LLM and case Elo ratings to all matches at once.

    matches: list of dicts with model_id, case_id, result (0 / 0.5 / 1).
    Maximum-likelihood logistic regression, solved by per-rating Newton
    steps until no rating moves more than tol Elo points. prior_weight is
    the strength of the one imaginary draw against a 1500 opponent that
    keeps undefeated (or winless) entities finite; it also pins the scale
    during the fit, so no re-anchoring happens inside the loop (that would
    fight the prior and never settle). The curvature is floored at its
    global bound so a nearly-saturated rating cannot make Newton overshoot
    and oscillate. When the fit is done, every rating is shifted equally -
    which changes no differences and therefore no probabilities - so the
    average case sits exactly at 1500 for reporting.

    Returns (llm_ratings, case_ratings, iterations).
    """
    by_llm = {}
    by_case = {}
    for match in matches:
        by_llm.setdefault(match["model_id"], []).append(match)
        by_case.setdefault(match["case_id"], []).append(match)
    llm_ratings = {m: ELO_CENTER for m in by_llm}
    case_ratings = {c: ELO_CENTER for c in by_case}

    iterations = 0
    for iterations in range(1, max_iter + 1):
        biggest_move = 0.0

        for model_id, games in by_llm.items():
            rating = llm_ratings[model_id]
            p0 = expected_win(rating, ELO_CENTER)
            gradient = prior_weight * (0.5 - p0)
            curvature = prior_weight * p0 * (1 - p0)
            for match in games:
                p = expected_win(rating, case_ratings[match["case_id"]])
                gradient += match["result"] - p
                curvature += p * (1 - p)
            curvature = max(curvature, (len(games) + prior_weight) / 8.0)
            step = max(-200.0, min(200.0, gradient / (curvature * ELO_SCALE)))
            llm_ratings[model_id] = rating + step
            biggest_move = max(biggest_move, abs(step))

        for case_id, games in by_case.items():
            rating = case_ratings[case_id]
            # From the case's side, the case "wins" what the LLM loses.
            p0 = expected_win(rating, ELO_CENTER)
            gradient = prior_weight * (0.5 - p0)
            curvature = prior_weight * p0 * (1 - p0)
            for match in games:
                p = expected_win(rating, llm_ratings[match["model_id"]])
                gradient += (1.0 - match["result"]) - p
                curvature += p * (1 - p)
            curvature = max(curvature, (len(games) + prior_weight) / 8.0)
            step = max(-200.0, min(200.0, gradient / (curvature * ELO_SCALE)))
            case_ratings[case_id] = rating + step
            biggest_move = max(biggest_move, abs(step))

        if biggest_move < tol:
            break

    # Cosmetic final anchor: average case = exactly 1500.
    if case_ratings:
        shift = ELO_CENTER - sum(case_ratings.values()) / len(case_ratings)
        for key in llm_ratings:
            llm_ratings[key] += shift
        for key in case_ratings:
            case_ratings[key] += shift

    return llm_ratings, case_ratings, iterations


# ---------- loading and joining the study files ----------


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_scores_files(folder="."):
    return sorted(
        name for name in os.listdir(folder) if SCORES_FILE_RE.match(name)
    )


def find_key_files(extra_folders=()):
    folders = [PACKAGES_DIR, "."]
    for extra in extra_folders:
        folders.extend([os.path.join(extra, PACKAGES_DIR), extra])
    found = []
    seen = set()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        real = os.path.realpath(folder)
        if real in seen:
            continue
        seen.add(real)
        for name in sorted(os.listdir(folder)):
            if KEY_FILE_RE.match(name):
                found.append(os.path.join(folder, name))
    return found


def load_keys(paths):
    """Returns (keys, rubric_versions):
    keys: package_id -> {case_id: {label: {model_id, model_reported}}}
    rubric_versions: package_id -> {case_id: version at packaging time}"""
    keys = {}
    rubric_versions = {}
    for path in paths:
        try:
            data = load_json_file(path)
        except (OSError, ValueError):
            print("Note: could not read key file {} (skipped).".format(path))
            continue
        package_id = data.get("package_id")
        if package_id and isinstance(data.get("key"), dict):
            keys[package_id] = data["key"]
            if isinstance(data.get("rubric_versions"), dict):
                rubric_versions[package_id] = data["rubric_versions"]
    return keys, rubric_versions


def load_scores(paths):
    loaded = []
    for path in paths:
        try:
            data = load_json_file(path)
        except (OSError, ValueError):
            print("Note: could not read scores file {} (skipped).".format(path))
            continue
        if not isinstance(data.get("scores"), list):
            print("Note: {} does not look like a scores file (skipped).".format(path))
            continue
        loaded.append(
            {
                "path": path,
                "package_id": data.get("package_id", ""),
                "package_name": data.get("package_name", ""),
                "scorer": data.get("scorer", ""),
                "updated_at": data.get("updated_at", ""),
                "records": data["scores"],
            }
        )
    return loaded


def dedupe_scores(scores_files):
    """Within ONE scorer and ONE package, each answer counts once: the
    most recently saved grade wins. This protects the ranking from a
    scorer's earlier scores file sitting next to their final one (or a
    stray copy of the same file) - identical evidence must not count
    twice, and superseded grades must not count at all. Grades from
    DIFFERENT scorers are deliberately all kept: more graders on the
    same answer is more evidence, and disagreement averages out inside
    the Elo fit.

    Returns (deduped_scores_files, notes)."""
    best = {}  # (package, scorer, case, label) -> (stamp, file_idx, record)
    dropped = 0
    for file_index, scores_file in enumerate(scores_files):
        for record in scores_file["records"]:
            key = (
                scores_file.get("package_id", ""),
                (scores_file.get("scorer") or "").strip().lower(),
                record.get("case_id"),
                record.get("label"),
            )
            stamp = (
                record.get("scored_at") or "",
                scores_file.get("updated_at") or "",
            )
            current = best.get(key)
            if current is None:
                best[key] = (stamp, file_index, record)
            elif stamp > current[0]:
                best[key] = (stamp, file_index, record)
                dropped += 1
            else:
                dropped += 1
    kept_by_file = {}
    for stamp, file_index, record in best.values():
        kept_by_file.setdefault(file_index, []).append(record)
    deduped = []
    for file_index, scores_file in enumerate(scores_files):
        copy = dict(scores_file)
        copy["records"] = kept_by_file.get(file_index, [])
        deduped.append(copy)
    notes = []
    if dropped:
        notes.append(
            "{} duplicate grade{} (same scorer, same answer) ignored - only "
            "the most recently saved grade counts. Grades from different "
            "scorers are all kept.".format(dropped, "" if dropped == 1 else "s")
        )
    return deduped, notes


def multi_scorer_summary(matches, names=None):
    """Plain-language inter-rater lines: how many answers were graded by
    more than one scorer, and how often the scorers agreed exactly."""
    names = names or {}
    groups = {}
    for match in matches:
        groups.setdefault((match["case_id"], match["model_id"]), []).append(match)
    multi = {pair: group for pair, group in groups.items() if len(group) > 1}
    if not multi:
        return []
    agreed = sum(
        1 for group in multi.values() if len({m["score"] for m in group}) == 1
    )
    lines = [
        "{} answer{} graded by more than one scorer; every scorer gave the "
        "same score on {} of {} ({:.0f}%).".format(
            len(multi), " was" if len(multi) == 1 else "s were",
            agreed, len(multi), 100.0 * agreed / len(multi),
        )
    ]
    disagreements = sorted(
        (pair, group) for pair, group in multi.items()
        if len({m["score"] for m in group}) > 1
    )
    for (case_id, model_id), group in disagreements[:8]:
        parts = ", ".join(
            "{} by {}".format(m["score"], m["scorer"] or "?")
            for m in sorted(group, key=lambda m: m["scorer"] or "")
        )
        lines.append("  Disagreement on case {} x {}: {}".format(
            case_id, names.get(model_id, model_id), parts
        ))
    if len(disagreements) > 8:
        lines.append("  (...and {} more disagreements)".format(len(disagreements) - 8))
    return lines


def reference_rubric_versions(scores_files, rubric_versions, current_versions):
    """The rubric version every grade of a case must match.

    The master database's current version wins when known. Otherwise the
    highest version seen anywhere (key file or any grade) is the
    reference - the key alone can be stale after a mid-study rubric fix,
    and re-graded answers carry the newer version.
    """
    reference = {}  # (package_id, case_id) -> version
    for scores_file in scores_files:
        package_id = scores_file["package_id"]
        packaged = (rubric_versions or {}).get(package_id) or {}
        for record in scores_file["records"]:
            case_id = record.get("case_id")
            spot = (package_id, case_id)
            candidates = [reference.get(spot, 1), record.get("rubric_version", 1)]
            if case_id in packaged:
                candidates.append(packaged[case_id])
            if (current_versions or {}).get(case_id):
                candidates = [current_versions[case_id]]
            reference[spot] = max(candidates)
    return reference


def build_matches(scores_files, keys, rubric_versions=None, current_versions=None):
    """Join grades to models. Returns (matches, warnings).

    Each graded answer becomes one match:
    {model_id, case_id, result, score, scorer, package_id}.

    Grades made under an outdated rubric version are refused: every model
    on a case must have been judged by the SAME rubric or the ratings are
    not comparable.
    """
    matches = []
    warnings = []
    reference = reference_rubric_versions(scores_files, rubric_versions, current_versions)
    for scores_file in scores_files:
        key = keys.get(scores_file["package_id"])
        if key is None:
            warnings.append(
                "{} belongs to package '{}' but no matching key file was found "
                "- its grades are left out. Key files live in {}/.".format(
                    scores_file["path"],
                    scores_file["package_name"] or scores_file["package_id"],
                    PACKAGES_DIR,
                )
            )
            continue
        for record in scores_file["records"]:
            case_id = record.get("case_id")
            label = record.get("label")
            score = record.get("score")
            entry = (key.get(case_id) or {}).get(label)
            if entry is None:
                warnings.append(
                    "{}: no key entry for case {} answer {} (left out).".format(
                        scores_file["path"], case_id, label
                    )
                )
                continue
            if score not in RESULT_FOR_SCORE:
                warnings.append(
                    "{}: case {} answer {} has no valid 0/1/2 score (left out).".format(
                        scores_file["path"], case_id, label
                    )
                )
                continue
            expected_version = reference.get((scores_file["package_id"], case_id), 1)
            grade_version = record.get("rubric_version", 1)
            if grade_version != expected_version:
                warnings.append(
                    "{}: case {} answer {} was graded under rubric version {} "
                    "but the study is on version {} - it must be re-graded "
                    "before it can be ranked (left out).".format(
                        scores_file["path"], case_id, label,
                        grade_version, expected_version,
                    )
                )
                continue
            matches.append(
                {
                    "model_id": entry.get("model_id", "unknown"),
                    "display": entry.get("display_name", ""),
                    "case_id": case_id,
                    "score": score,
                    "result": RESULT_FOR_SCORE[score],
                    "scorer": scores_file["scorer"],
                    "package_id": scores_file["package_id"],
                }
            )
    return matches, warnings


def display_map(matches):
    """model_id -> friendly name: what the key files recorded (LLM plus
    model name), falling back to the runner's registries for old data."""
    names = display_names()
    result = {}
    for match in matches:
        if match.get("display"):
            result.setdefault(match["model_id"], match["display"])
    for match in matches:
        model_id = match["model_id"]
        result.setdefault(model_id, names.get(model_id, model_id))
    return result


def display_names():
    """model_id -> friendly name, from the runner's registries."""
    names = {"testmodel": "Test model (fake)"}
    try:
        from llm_api import API_REGISTRY

        for model_id, entry in API_REGISTRY.items():
            names[model_id] = entry["display_name"]
    except ImportError:
        pass
    try:
        from llm_browser import SITE_INFO

        for site_id, entry in SITE_INFO.items():
            names[site_id] = entry["display_name"]
    except ImportError:
        pass
    return names


# ---------- reports ----------


def summarize(matches):
    per_model = {}
    for match in matches:
        stats = per_model.setdefault(
            match["model_id"], {"n": 0, "sum": 0, "counts": {0: 0, 1: 0, 2: 0}}
        )
        stats["n"] += 1
        stats["sum"] += match["score"]
        stats["counts"][match["score"]] += 1
    return per_model


def export_csv(matches, llm_ratings, case_ratings):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    names = display_map(matches)
    per_model = summarize(matches)

    rankings_path = os.path.join(RESULTS_DIR, "llm_rankings.csv")
    with open(rankings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model_id", "model_name", "elo", "answers_graded",
                         "average_score", "score_2_count", "score_1_count", "score_0_count"])
        ranked = sorted(llm_ratings.items(), key=lambda item: -item[1])
        for rank, (model_id, rating) in enumerate(ranked, start=1):
            stats = per_model[model_id]
            writer.writerow([
                rank, model_id, names.get(model_id, model_id), round(rating, 1),
                stats["n"], round(stats["sum"] / stats["n"], 3),
                stats["counts"][2], stats["counts"][1], stats["counts"][0],
            ])

    cases_path = os.path.join(RESULTS_DIR, "case_difficulty.csv")
    with open(cases_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "provider_number", "elo", "answers_graded", "average_score"])
        per_case = {}
        for match in matches:
            stats = per_case.setdefault(match["case_id"], {"n": 0, "sum": 0})
            stats["n"] += 1
            stats["sum"] += match["score"]
        for case_id, rating in sorted(case_ratings.items(), key=lambda item: -item[1]):
            stats = per_case[case_id]
            writer.writerow([
                case_id, split_case_id(case_id)[0], round(rating, 1),
                stats["n"], round(stats["sum"] / stats["n"], 3),
            ])

    matches_path = os.path.join(RESULTS_DIR, "matches.csv")
    with open(matches_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "provider_number", "model_id", "model_name",
                         "score", "result_for_llm", "scorer", "package_id"])
        for match in sorted(matches, key=lambda m: (split_case_id(m["case_id"]), m["model_id"])):
            writer.writerow([
                match["case_id"], split_case_id(match["case_id"])[0],
                match["model_id"], names.get(match["model_id"], match["model_id"]),
                match["score"], match["result"], match["scorer"], match["package_id"],
            ])

    return [rankings_path, cases_path, matches_path]


# ---------- window interface ----------


def choose_scores_folder(root, filedialog, messagebox):
    """Ask where the scorers' returned scores files are; retry until a
    folder containing at least one is chosen. Returns the folder or None."""
    start_dir = os.getcwd()
    while True:
        folder = filedialog.askdirectory(
            parent=root,
            title="Where are the scores files the scorers emailed back? "
            "Choose the folder containing scores_*.json",
            initialdir=start_dir,
        )
        if not folder:
            # Cancelled: fall back to the study folder when it has some.
            if find_scores_files(start_dir):
                return start_dir
            messagebox.showinfo(
                "No folder chosen",
                "To rank the LLMs, start the program again and choose the "
                "folder where you saved the scores files the scorers emailed "
                "back (scores_*.json).",
            )
            return None
        if find_scores_files(folder):
            return folder
        if not messagebox.askretrycancel(
            "No scores files there",
            "No scores files (scores_*.json) were found in:\n{}\n\nChoose "
            "the folder where you saved the files the scorers emailed "
            "back.".format(folder),
        ):
            return None
        start_dir = folder


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import gui_common

    root = gui_common.make_root("LLM Ranker (for the PI)", 1000, 640)
    root.withdraw()

    scores_folder = choose_scores_folder(root, filedialog, messagebox)
    if scores_folder is None:
        root.destroy()
        return 1
    scores_files = load_scores(
        [os.path.join(scores_folder, name) for name in find_scores_files(scores_folder)]
    )
    keys, packaged_versions = load_keys(find_key_files(extra_folders=[scores_folder]))
    # The master database, when present, is the authority on the current
    # rubric version of every case.
    current_versions = {}
    try:
        from merge_cases import MASTER_FILENAME, MasterStore

        if os.path.exists(MASTER_FILENAME):
            master = MasterStore.load(MASTER_FILENAME)
            current_versions = {
                case_id: case.get("rubric_version", 1)
                for case_id, case in master.cases.items()
            }
    except Exception:
        current_versions = {}
    if not scores_files:
        gui_common.show_error(
            "No scores files",
            "The scores files in {} could not be read.".format(scores_folder),
        )
        root.destroy()
        return 1
    if not keys:
        gui_common.show_error(
            "No key files",
            "No key files (*_KEY_DO_NOT_SEND.json) were found here or in {}/.\n\n"
            "The ranker needs them to know which AI wrote each blinded "
            "answer.".format(PACKAGES_DIR),
        )
        root.destroy()
        return 1

    scores_files, dedupe_notes = dedupe_scores(scores_files)
    matches, warnings = build_matches(
        scores_files, keys, packaged_versions, current_versions
    )
    warnings = dedupe_notes + warnings
    if not matches:
        gui_common.show_error(
            "Nothing to rank",
            "No grades could be joined to a key file.\n\n" + "\n".join(warnings[:8]),
        )
        root.destroy()
        return 1

    llm_ratings, case_ratings, iterations = fit_ratings(matches)
    names = display_map(matches)
    per_model = summarize(matches)

    llms = {m["model_id"] for m in matches}
    cases = {m["case_id"] for m in matches}
    summary = (
        "{} scores file(s), {} graded answers usable: {} LLM(s) across {} case(s). "
        "Ratings fitted to all matches at once (logistic regression, {} passes)."
    ).format(len(scores_files), len(matches), len(llms), len(cases), iterations)
    agreement_lines = multi_scorer_summary(matches, names)
    if agreement_lines:
        summary += "\n" + agreement_lines[0]

    tk.Label(root, text=summary, anchor="w", justify="left", wraplength=960).pack(
        fill="x", padx=8, pady=(8, 0)
    )
    warnings = warnings + agreement_lines[1:]
    if warnings:
        warn_box = gui_common.LogBox(root, height=min(4, len(warnings)))
        warn_box.pack(fill="x", padx=8, pady=(4, 0))
        for warning in warnings:
            warn_box.log("Warning: " + warning)

    from tkinter import ttk

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    rank_frame = tk.Frame(notebook)
    notebook.add(rank_frame, text="LLM rankings")
    tk.Label(
        rank_frame,
        text="Higher Elo = stronger. The average case is 1500; a score of 2 is a "
        "win for the LLM, 1 a draw, 0 a loss.",
        anchor="w",
    ).pack(fill="x", pady=(6, 0))
    rank_tree = gui_common.make_table(
        rank_frame,
        [("rank", "Rank"), ("name", "LLM"), ("elo", "Elo"), ("n", "Graded"),
         ("avg", "Avg score"), ("counts", "2s / 1s / 0s"), ("p", "Predicted win vs avg case")],
        widths={"rank": 50, "name": 200, "elo": 70, "n": 70, "avg": 80,
                "counts": 110, "p": 170},
    )
    ranked = sorted(llm_ratings.items(), key=lambda item: -item[1])
    for rank, (model_id, rating) in enumerate(ranked, start=1):
        stats = per_model[model_id]
        rank_tree.insert("", "end", values=(
            rank, names.get(model_id, model_id), "{:.0f}".format(rating), stats["n"],
            "{:.2f}".format(stats["sum"] / stats["n"]),
            "{} / {} / {}".format(stats["counts"][2], stats["counts"][1], stats["counts"][0]),
            "{:.0f}%".format(100 * expected_win(rating, ELO_CENTER)),
        ))
    rank_tree.master.pack(fill="both", expand=True, pady=6)

    case_frame = tk.Frame(notebook)
    notebook.add(case_frame, text="Case difficulty")
    tk.Label(
        case_frame,
        text="Higher Elo = harder case (the AIs scored worse on it).", anchor="w",
    ).pack(fill="x", pady=(6, 0))
    case_tree = gui_common.make_table(
        case_frame,
        [("case", "Case"), ("elo", "Elo"), ("n", "Graded"), ("avg", "Avg score")],
        widths={"case": 100, "elo": 80, "n": 80, "avg": 90},
    )
    per_case = {}
    for m in matches:
        stats = per_case.setdefault(m["case_id"], {"n": 0, "sum": 0})
        stats["n"] += 1
        stats["sum"] += m["score"]
    for case_id, rating in sorted(case_ratings.items(), key=lambda item: -item[1]):
        stats = per_case[case_id]
        case_tree.insert("", "end", values=(
            case_id, "{:.0f}".format(rating), stats["n"],
            "{:.2f}".format(stats["sum"] / stats["n"]),
        ))
    case_tree.master.pack(fill="both", expand=True, pady=6)

    bottom = tk.Frame(root)
    bottom.pack(fill="x", padx=8, pady=(0, 8))
    status = tk.Label(bottom, text="", anchor="w")

    def do_export():
        try:
            paths = export_csv(matches, llm_ratings, case_ratings)
        except OSError as error:
            messagebox.showerror("Could not write the files", str(error), parent=root)
            return
        status.configure(
            text="Wrote {} (rankings, case difficulty, and every match).".format(
                ", ".join(paths)
            )
        )

    tk.Button(bottom, text="Export CSV files for analysis", command=do_export).pack(side="left")
    status.pack(side="left", fill="x", expand=True, padx=8)

    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
