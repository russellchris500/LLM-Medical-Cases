#!/usr/bin/env python3
"""Program 3 of the LLM Medical Cases evaluation framework: the LLM Runner.

The principal investigator (PI) uses this tool to run cases from the master
database (master_cases.json, built by Program 2) through a chosen set of
LLMs, and to store every answer - text and images - in answers.json and
answer_images/ for later scoring.

Everything is menu-driven - just run:  python3 run_llms.py

Models:
- API models (Anthropic Claude, OpenAI GPT, Google Gemini, xAI Grok) run
  unattended once an API key is entered in Settings.
- Browser models (OpenEvidence, UpToDate, Doximity GPT) are driven through
  a visible browser window; the PI should stay at the computer for that
  part in case a site asks for a login or verification.

Requires Python 3.8+. The API models need nothing installed; the browser
models need Playwright, which the Settings menu can install for you.
"""

import getpass
import os
import subprocess
import sys
import time
import random

from case_editor import CaseStoreError, now_iso, prompt
from merge_cases import MASTER_FILENAME, MasterStore
from eval_common import (
    AnswersStore,
    CaseSetStore,
    OK_STATUSES,
    SelectionError,
    SettingsStore,
    case_hash,
    parse_selection,
    sort_case_ids,
    split_case_id,
)
from llm_api import API_MODEL_IDS, API_REGISTRY, ApiCallError, ModelAbort, call_api_model
import llm_browser
from llm_browser import (
    BROWSER_MODEL_IDS,
    BrowserStepError,
    SITE_INFO,
    copy_to_clipboard,
    make_driver,
    open_site_context,
)

PROMPT_TEMPLATE = (
    "You are a physician being asked about a clinical case. Read the case "
    "and give your assessment and recommendations.\n\n{case_text}"
)
PROMPT_TEMPLATE_VERSION = 1

TEST_MODEL_ID = "testmodel"


class AbandonSite(Exception):
    """The user gave up on one browser site for this run."""


def render_prompt(case):
    return PROMPT_TEMPLATE.format(case_text=case["case_text"])


def model_catalog(settings):
    """Every model the runner knows, in menu order."""
    catalog = []
    for model_id in API_MODEL_IDS:
        catalog.append(
            {
                "model_id": model_id,
                "display_name": API_REGISTRY[model_id]["display_name"],
                "kind": "api",
            }
        )
    for site_id in BROWSER_MODEL_IDS:
        catalog.append(
            {
                "model_id": site_id,
                "display_name": SITE_INFO[site_id]["display_name"],
                "kind": "browser",
            }
        )
    if settings.option("enable_test_model"):
        catalog.append(
            {"model_id": TEST_MODEL_ID, "display_name": "Test model (fake)", "kind": "test"}
        )
    return catalog


def new_record(case, model, prompt_sent):
    return {
        "case_id": case["case_id"],
        "model_id": model["model_id"],
        "model_kind": model["kind"],
        "model_display_name": model["display_name"],
        "model_requested": "",
        "model_reported": "",
        "prompt_sent": prompt_sent,
        "response_text": "",
        "images": [],
        "status": "failed",
        "error": None,
        "case_sha256": case_hash(case),
        "case_updated_at": case.get("updated_at"),
        "started_at": now_iso(),
        "finished_at": None,
        "attempts": 1,
    }


# ---------- case selection ----------


def describe_selection(case_ids, limit=8):
    if len(case_ids) <= limit:
        return ", ".join(case_ids)
    return "{} ... {} ({} in all)".format(
        ", ".join(case_ids[:3]), ", ".join(case_ids[-2:]), len(case_ids)
    )


def choose_saved_set(case_sets, cases_by_id):
    if not case_sets.sets:
        print("There are no saved case sets yet.")
        return None
    names = sorted(case_sets.sets, key=str.lower)
    print("Saved case sets:")
    for i, name in enumerate(names, start=1):
        entry = case_sets.sets[name]
        print(
            "  {}. {}  ({} cases; {})".format(
                i, name, len(entry["case_ids"]), entry["expression"] or "chosen by hand"
            )
        )
    raw = prompt("Which set? Enter its number (blank to cancel): ").strip()
    if not raw:
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= len(names):
        print("There is no set number {}.".format(raw))
        return None
    name = names[int(raw) - 1]
    present, missing = case_sets.resolve(name, cases_by_id)
    if missing:
        print(
            "{} case{} in this set {} no longer in the master and will be "
            "skipped: {}".format(
                len(missing),
                "" if len(missing) == 1 else "s",
                "is" if len(missing) == 1 else "are",
                ", ".join(missing),
            )
        )
    if not present:
        print("None of this set's cases are in the master database.")
        return None
    return present


def offer_to_save_set(case_sets, expression, case_ids):
    name = prompt(
        "Save this selection as a named set for reuse? Name (blank for no): "
    ).strip()
    if not name:
        return
    try:
        case_sets.add(name, expression, case_ids)
        print("Saved case set '{}' ({} cases).".format(name, len(case_ids)))
    except CaseStoreError as e:
        print("Not saved: {}".format(e))


def choose_cases(master, case_sets, header="Step 1 of 3 - choose the cases."):
    cases_by_id = master.cases
    providers = sorted({c["provider_number"] for c in cases_by_id.values()})
    while True:
        print("\n{}".format(header))
        print("  [A]ll {} cases".format(len(cases_by_id)))
        print("  [E]nter case IDs or ranges (e.g. 003-001..003-020, 005-004)")
        print("  [P] all cases from one provider")
        print("  [S]aved case set                     ({} saved)".format(len(case_sets.sets)))
        choice = prompt("Choose A, E, P, or S (blank to cancel): ").strip().lower()
        if not choice:
            return None
        if choice == "a":
            return sort_case_ids(cases_by_id)
        if choice == "s":
            selected = choose_saved_set(case_sets, cases_by_id)
            if selected:
                return selected
            continue
        if choice == "p":
            print("Providers with cases: {}".format(", ".join(str(p) for p in providers)))
            raw = prompt("Which provider number? ").strip()
            if not raw.isdigit():
                continue
            expression = "provider {}".format(int(raw))
        elif choice == "e":
            expression = prompt(
                "Enter cases (comma-separated; ranges like 003-001..003-020): "
            ).strip()
            if not expression:
                continue
        else:
            print("Please choose A, E, P, or S.")
            continue

        try:
            selected, warnings = parse_selection(expression, cases_by_id)
        except SelectionError as e:
            print(str(e))
            continue
        for warning in warnings:
            print("  Note: {}".format(warning))
        if not selected:
            print("That selection matched no cases.")
            continue
        print("Selected {} case{}: {}".format(
            len(selected), "" if len(selected) == 1 else "s", describe_selection(selected)
        ))
        offer_to_save_set(case_sets, expression, selected)
        return selected


# ---------- model selection ----------


def model_status_line(model, settings, answers, selected_cases):
    parts = []
    if model["kind"] == "api":
        key = settings.api_model(model["model_id"]).get("api_key", "")
        parts.append("API, key set" if key.strip() else "API, NO KEY - set it in Settings")
    elif model["kind"] == "browser":
        if not llm_browser.PLAYWRIGHT_AVAILABLE:
            parts.append("browser, needs one-time setup (Settings)")
        else:
            last = settings.browser_model(model["model_id"]).get("last_login_ok")
            parts.append(
                "browser, login OK {}".format(last[:10]) if last else "browser, never logged in"
            )
    else:
        parts.append("fake test model")
    answered = sum(
        1
        for case_id in selected_cases
        if (answers.get(case_id, model["model_id"]) or {}).get("status") in OK_STATUSES
    )
    parts.append("answered {}/{} of these".format(answered, len(selected_cases)))
    return "; ".join(parts)


def choose_models(settings, answers, selected_cases):
    catalog = model_catalog(settings)
    print("\nStep 2 of 3 - choose the LLMs.")
    for i, model in enumerate(catalog, start=1):
        print(
            "  {}. {:18s} ({})".format(
                i, model["display_name"], model_status_line(model, settings, answers, selected_cases)
            )
        )
    raw = prompt("Enter numbers (e.g. 2,4,5) or A for all (blank to cancel): ").strip()
    if not raw:
        return None
    if raw.lower() in ("a", "all"):
        chosen = list(catalog)
    else:
        chosen = []
        for part in raw.replace(",", " ").split():
            if not part.isdigit() or not 1 <= int(part) <= len(catalog):
                print("There is no model number {}.".format(part))
                return None
            model = catalog[int(part) - 1]
            if model not in chosen:
                chosen.append(model)

    usable = []
    for model in chosen:
        if model["kind"] == "api" and not settings.api_model(model["model_id"]).get("api_key", "").strip():
            print(
                "Skipping {}: no API key. Add it in Settings first.".format(model["display_name"])
            )
        elif model["kind"] == "browser" and not llm_browser.PLAYWRIGHT_AVAILABLE:
            print(
                "Skipping {}: browser automation is not set up yet. Use "
                "'Browser automation setup' in Settings first.".format(model["display_name"])
            )
        else:
            usable.append(model)
    return usable or None


# ---------- run confirmation ----------


def build_worklist(master, answers, case_ids, models):
    """Decide which (case, model) pairs actually need asking."""
    todo = {model["model_id"]: [] for model in models}
    skipped = 0
    failed_pairs = []
    changed_pairs = []
    for model in models:
        for case_id in case_ids:
            existing = answers.get(case_id, model["model_id"])
            if existing is None:
                todo[model["model_id"]].append(case_id)
            elif existing.get("status") in OK_STATUSES:
                if existing.get("case_sha256") != case_hash(master.cases[case_id]):
                    changed_pairs.append((case_id, model))
                else:
                    skipped += 1
            else:
                failed_pairs.append((case_id, model))
    return todo, skipped, failed_pairs, changed_pairs


def confirm_run(master, answers, case_ids, models):
    todo, skipped, failed_pairs, changed_pairs = build_worklist(
        master, answers, case_ids, models
    )
    total_requested = len(case_ids) * len(models)
    print("\nStep 3 of 3 - confirm.")
    print(
        "  {} case-model pairs requested; {} already answered (will skip).".format(
            total_requested, skipped
        )
    )
    if failed_pairs:
        if prompt(
            "  {} pair{} failed before. Retry {}? [Y/n]: ".format(
                len(failed_pairs),
                "" if len(failed_pairs) == 1 else "s",
                "it" if len(failed_pairs) == 1 else "them",
            )
        ).strip().lower() in ("", "y", "yes"):
            for case_id, model in failed_pairs:
                todo[model["model_id"]].append(case_id)
    if changed_pairs:
        print(
            "  {} answered pair{} where the case wording has changed since:".format(
                len(changed_pairs), "" if len(changed_pairs) == 1 else "s"
            )
        )
        for case_id, model in changed_pairs[:10]:
            print("    {} x {}".format(case_id, model["display_name"]))
        if prompt("  Ask these again (overwrites the old answers)? [y/N]: ").strip().lower() in (
            "y",
            "yes",
        ):
            for case_id, model in changed_pairs:
                todo[model["model_id"]].append(case_id)

    for model_id in todo:
        todo[model_id] = sort_case_ids(set(todo[model_id]))

    api_count = sum(len(todo[m["model_id"]]) for m in models if m["kind"] in ("api", "test"))
    browser_models = [m for m in models if m["kind"] == "browser" and todo[m["model_id"]]]
    browser_count = sum(len(todo[m["model_id"]]) for m in browser_models)
    if api_count + browser_count == 0:
        print("  Nothing to do - everything selected is already answered.")
        return None
    line = "  {} answer{} to collect: {} by API (unattended)".format(
        api_count + browser_count, "" if api_count + browser_count == 1 else "s", api_count
    )
    if browser_models:
        line += ", {} via {} (browser - please stay at the computer for that part)".format(
            browser_count, ", ".join(m["display_name"] for m in browser_models)
        )
    print(line + ".")
    if prompt("Start? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        return None
    return todo


# ---------- running: API + test models ----------


def run_test_model(case):
    return {
        "response_text": "TEST ANSWER for {}: this canned reply comes from the "
        "built-in fake model used to try the programs end to end.".format(case["case_id"]),
        "model_requested": "test-model-1",
        "model_reported": "test-model-1",
        "attempts": 1,
    }


def run_api_phase(master, answers, settings, models, todo):
    api_models = [m for m in models if m["kind"] in ("api", "test") and todo[m["model_id"]]]
    if not api_models:
        return
    total = sum(len(todo[m["model_id"]]) for m in api_models)
    print("\nAPI models:")
    done = 0
    options = settings.data["options"]
    for model in api_models:
        for case_id in todo[model["model_id"]]:
            done += 1
            case = master.cases[case_id]
            label = "  [{:3d}/{}] {} x {} ".format(done, total, case_id, model["display_name"])
            print(label.ljust(46, "."), end=" ", flush=True)
            record = new_record(case, model, render_prompt(case))
            started = time.monotonic()
            try:
                if model["kind"] == "test":
                    result = run_test_model(case)
                else:
                    result = call_api_model(
                        model["model_id"],
                        settings.api_model(model["model_id"]),
                        record["prompt_sent"],
                        options,
                    )
                record.update(result)
                record["status"] = "ok"
                record["finished_at"] = now_iso()
                answers.upsert(record)
                print("ok ({:.1f}s)".format(time.monotonic() - started))
            except ApiCallError as e:
                record["error"] = str(e)
                record["finished_at"] = now_iso()
                answers.upsert(record)
                print("FAILED - {}".format(e))
            except ModelAbort as e:
                print("stopped")
                print("  {} is being skipped for the rest of this run: {}".format(
                    model["display_name"], e
                ))
                break


# ---------- running: browser models ----------


def manual_capture(driver, page, case, prompt_text, images_dir, basename):
    if copy_to_clipboard(page, prompt_text):
        print(
            "  The case text is on your clipboard. In the browser window: paste "
            "it (Ctrl+V / Cmd+V), send it, and wait for the full answer."
        )
    else:
        print("  Copy the case text between the lines below into the site:")
        print("-" * 60)
        print(prompt_text)
        print("-" * 60)
    prompt("  When the answer is fully visible, press Enter here to capture it. ")
    return driver.extract_answer(page, images_dir, basename, manual=True)


def browser_ask_one(driver, page, case, prompt_text, images_dir, basename, options):
    driver.start_new_question(page)
    baseline = driver.baseline_text(page)
    driver.submit_question(page, prompt_text)
    driver.wait_for_answer(
        page,
        baseline,
        stable_seconds=options.get("answer_stable_seconds", 10),
        max_wait_seconds=options.get("answer_max_wait_seconds", 300),
    )
    return driver.extract_answer(page, images_dir, basename, baseline=baseline)


def browser_recovery_menu(driver, site_name, case_id, error):
    print(
        "\n  Problem on {} with case {}: {}.".format(site_name, case_id, error.detail)
    )
    print("  The browser window is still open.")
    print("    [R]etry automatically")
    print("    [M] do it by hand (the program will capture the result)")
    print("    [S]kip this case on {}".format(site_name))
    print("    [A]bandon {} for this run (other models are unaffected)".format(site_name))
    if error.step == "logged_out":
        print("    [L] I have logged back in - continue")
    while True:
        choice = prompt("  > ").strip().lower()
        if choice in ("r", "m", "s", "a") or (choice == "l" and error.step == "logged_out"):
            return choice
        print("  Please choose one of the letters above.")


def interactive_login(driver, page, site_settings):
    try:
        page.goto(driver.login_url, wait_until="domcontentloaded")
    except Exception:
        pass
    driver.autofill_login(
        page, site_settings.get("username", ""), site_settings.get("password", "")
    )
    print(
        "\n  A browser window is open on {}. Please finish logging in there,\n"
        "  including any verification code or \"I am not a robot\" check.\n"
        "  If the site offers \"remember this device\", say yes.".format(driver.display_name)
    )
    while True:
        raw = prompt(
            "  When you can see the normal question page, press Enter (or S to stop): "
        ).strip().lower()
        if raw == "s":
            return False
        try:
            page.goto(driver.home_url, wait_until="domcontentloaded")
        except Exception:
            pass
        if driver.is_logged_in(page):
            site_settings["last_login_ok"] = now_iso()
            return True
        print("  It doesn't look logged in yet - please finish in the browser window.")


def run_browser_site(master, answers, settings, model, case_ids):
    site_id = model["model_id"]
    options = settings.data["options"]
    print(
        "\n  Next: {} ({} case{}). A browser window will open; please stay at\n"
        "  the computer in case the site asks you to log in or verify.".format(
            model["display_name"], len(case_ids), "" if len(case_ids) == 1 else "s"
        )
    )
    raw = prompt(
        "  Press Enter to begin, M to answer every case by hand in the browser,\n"
        "  or S to skip {} for now: ".format(model["display_name"])
    ).strip().lower()
    if raw == "s":
        return
    all_manual = raw == "m"

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        context = open_site_context(playwright, site_id)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            driver = make_driver(site_id)
            driver.start_new_question(page)
            if not driver.is_logged_in(page):
                if not interactive_login(driver, page, settings.browser_model(site_id)):
                    print("  Skipping {} (not logged in).".format(model["display_name"]))
                    return
                settings.save()

            images_dir = answers.ensure_images_dir()
            for index, case_id in enumerate(case_ids, start=1):
                case = master.cases[case_id]
                prompt_text = render_prompt(case)
                basename = answers.image_basename(case_id, site_id)
                record = new_record(case, model, prompt_text)
                started = time.monotonic()
                print(
                    "  [{:3d}/{}] {} x {}".format(
                        index, len(case_ids), case_id, model["display_name"]
                    )
                )
                answers.clear_images(case_id, site_id)
                result = None
                mode_manual = all_manual
                while result is None:
                    try:
                        if mode_manual:
                            result = manual_capture(
                                driver, page, case, prompt_text, images_dir, basename
                            )
                        else:
                            result = browser_ask_one(
                                driver, page, case, prompt_text, images_dir, basename, options
                            )
                    except BrowserStepError as error:
                        choice = browser_recovery_menu(
                            driver, model["display_name"], case_id, error
                        )
                        if choice == "r" or choice == "l":
                            continue
                        if choice == "m":
                            mode_manual = True
                            continue
                        if choice == "s":
                            record["error"] = "skipped: " + error.detail
                            record["finished_at"] = now_iso()
                            answers.upsert(record)
                            break
                        if choice == "a":
                            raise AbandonSite()
                if result is None:
                    continue
                record["response_text"] = result.text
                record["images"] = [p.replace(os.sep, "/") for p in result.image_paths]
                record["model_requested"] = model["display_name"]
                record["model_reported"] = result.model_reported
                record["status"] = "ok_manual" if result.manual else "ok"
                record["finished_at"] = now_iso()
                answers.upsert(record)
                print(
                    "        ok{}, {} image{} ({:.0f}s)".format(
                        " (by hand)" if result.manual else "",
                        len(result.image_paths),
                        "" if len(result.image_paths) == 1 else "s",
                        time.monotonic() - started,
                    )
                )
                if index < len(case_ids) and not mode_manual:
                    delay = options.get("browser_question_delay_s", 8)
                    time.sleep(delay * random.uniform(0.5, 1.5))
        except AbandonSite:
            print("  Stopped {} for this run.".format(model["display_name"]))
        finally:
            try:
                context.close()
            except Exception:
                pass


def run_browser_phase(master, answers, settings, models, todo):
    browser_models = [m for m in models if m["kind"] == "browser" and todo[m["model_id"]]]
    if not browser_models:
        return
    print("\nBrowser models:")
    for model in browser_models:
        run_browser_site(master, answers, settings, model, todo[model["model_id"]])


def run_flow(master, answers, settings, case_sets):
    case_ids = choose_cases(master, case_sets)
    if not case_ids:
        return
    models = choose_models(settings, answers, case_ids)
    if not models:
        return
    todo = confirm_run(master, answers, case_ids, models)
    if todo is None:
        return
    try:
        run_api_phase(master, answers, settings, models, todo)
        run_browser_phase(master, answers, settings, models, todo)
    except KeyboardInterrupt:
        print(
            "\nStopped. Everything answered so far is saved in answers.json; "
            "run again to continue where you left off."
        )
        return
    summarize_run(answers, case_ids, models)


def summarize_run(answers, case_ids, models):
    ok = manual = failed = missing = 0
    failures = []
    for model in models:
        for case_id in case_ids:
            record = answers.get(case_id, model["model_id"])
            if record is None:
                missing += 1
            elif record["status"] == "ok":
                ok += 1
            elif record["status"] == "ok_manual":
                manual += 1
            else:
                failed += 1
                failures.append(
                    "{} x {}: {}".format(case_id, model["display_name"], record.get("error"))
                )
    line = "\nDone. {} ok".format(ok)
    if manual:
        line += ", {} ok (by hand)".format(manual)
    if failed:
        line += ", {} failed".format(failed)
    if missing:
        line += ", {} not attempted".format(missing)
    print(line + ".")
    for failure in failures[:10]:
        print("  " + failure)
    print(
        "Everything is saved after each answer - you can re-run anytime; "
        "finished pairs are skipped automatically."
    )


# ---------- case sets menu ----------


def case_sets_menu(master, case_sets):
    while True:
        names = sorted(case_sets.sets, key=str.lower)
        print("\nSaved case sets ({}):".format(len(names)))
        for name in names:
            entry = case_sets.sets[name]
            print(
                "  {}  ({} cases; {})".format(
                    name, len(entry["case_ids"]), entry["expression"] or "chosen by hand"
                )
            )
        choice = prompt(
            "[N]ew set  [V]iew  [R]ename  [D]elete  or press Enter to go back: "
        ).strip().lower()
        if not choice:
            return
        try:
            if choice == "n":
                selected = choose_cases(master, case_sets, header="Choose the cases for the new set.")
                if selected is None:
                    continue
                # choose_cases already offered to save ad-hoc entries; only
                # ask again if it wasn't saved there.
                if not any(case_sets.sets[n]["case_ids"] == selected for n in case_sets.sets):
                    name = prompt("Name for this set: ").strip()
                    if name:
                        case_sets.add(name, "chosen by hand", selected)
                        print("Saved case set '{}'.".format(name))
            elif choice in ("v", "r", "d"):
                name = prompt("Which set name? ").strip()
                stored = case_sets.find(name)
                if stored is None:
                    print("No case set named '{}'.".format(name))
                    continue
                if choice == "v":
                    entry = case_sets.sets[stored]
                    present, missing = case_sets.resolve(stored, master.cases)
                    print("Set '{}' - {}".format(stored, entry["expression"]))
                    print("  Cases: {}".format(describe_selection(entry["case_ids"], limit=30)))
                    if missing:
                        print("  No longer in the master: {}".format(", ".join(missing)))
                elif choice == "r":
                    new_name = prompt("New name: ").strip()
                    if new_name:
                        case_sets.rename(stored, new_name)
                        print("Renamed to '{}'.".format(new_name))
                else:
                    if prompt(
                        "Really delete set '{}'? Type yes to confirm: ".format(stored)
                    ).strip().lower() == "yes":
                        case_sets.delete(stored)
                        print("Deleted.")
            else:
                print("Please choose N, V, R, or D.")
        except CaseStoreError as e:
            print("Error: {}".format(e))


# ---------- answers menu ----------


def answers_menu(master, answers, settings):
    if not answers.answers:
        print("\nNo answers have been collected yet.\n")
        return
    catalog = {m["model_id"]: m for m in model_catalog(settings)}
    model_ids = answers.model_ids()
    providers = sorted({split_case_id(c)[0] for c, _ in answers.answers})
    print("\nAnswers collected so far (ok / failed):")
    header = "  {:22s}".format("")
    for provider in providers:
        header += "  provider {:>4}".format(provider)
    print(header)
    for model_id in model_ids:
        display = catalog.get(model_id, {}).get("display_name", model_id)
        row = "  {:22s}".format(display[:22])
        for provider in providers:
            ok = failed = 0
            for (case_id, mid), record in answers.answers.items():
                if mid != model_id or split_case_id(case_id)[0] != provider:
                    continue
                if record["status"] in OK_STATUSES:
                    ok += 1
                else:
                    failed += 1
            row += "  {:>8}".format("{} / {}".format(ok, failed))
        print(row)

    failures = [
        (case_id, mid, record)
        for (case_id, mid), record in sorted(answers.answers.items())
        if record["status"] not in OK_STATUSES
    ]
    while True:
        choice = prompt(
            "\n[V]iew one answer  [F] list failed pairs  or press Enter to go back: "
        ).strip().lower()
        if not choice:
            return
        if choice == "f":
            if not failures:
                print("No failed pairs.")
            for case_id, mid, record in failures:
                print("  {} x {}: {}".format(case_id, mid, record.get("error")))
        elif choice == "v":
            case_id = prompt("Case ID (e.g. 003-001): ").strip()
            mid = prompt(
                "Model ({}): ".format(", ".join(model_ids))
            ).strip().lower()
            record = answers.get(case_id, mid)
            if record is None:
                print("No answer stored for {} x {}.".format(case_id, mid))
                continue
            print("\n{} x {}  [{}]  asked {}".format(
                case_id, record.get("model_display_name", mid), record["status"],
                record.get("started_at"),
            ))
            print("Model: {}".format(record.get("model_reported") or record.get("model_requested")))
            print("-" * 60)
            print(record.get("response_text") or "(no text; error: {})".format(record.get("error")))
            print("-" * 60)
            if record.get("images"):
                print("Images: {}".format(", ".join(record["images"])))
        else:
            print("Please choose V or F.")


# ---------- settings menu ----------


def masked(secret):
    secret = (secret or "").strip()
    if not secret:
        return "not set"
    return "..." + secret[-4:]


def enter_secret(label):
    try:
        value = getpass.getpass("{} (typing is hidden; blank to keep current): ".format(label))
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return value.strip() or None


def test_api_connection(model_id, settings):
    entry = settings.api_model(model_id)
    print("Contacting {}...".format(API_REGISTRY[model_id]["display_name"]))
    options = dict(settings.data["options"])
    options["max_retries"] = 1
    try:
        result = call_api_model(
            model_id, entry, "Reply with the single word OK.", options
        )
        print(
            "Success. The service answered and reported model '{}'.".format(
                result["model_reported"]
            )
        )
    except (ApiCallError, ModelAbort) as e:
        print("Test failed: {}".format(e))


def api_model_settings(model_id, settings):
    registry = API_REGISTRY[model_id]
    while True:
        entry = settings.api_model(model_id)
        print("\n{} settings:".format(registry["display_name"]))
        print("  Key: {}   Model: {}".format(
            masked(entry.get("api_key")),
            entry.get("model") or "(default: {})".format(registry["default_model"]),
        ))
        choice = prompt(
            "[K]ey  [M]odel name  [T]est connection  or press Enter to go back: "
        ).strip().lower()
        if not choice:
            return
        if choice == "k":
            value = enter_secret("API key for {}".format(registry["display_name"]))
            if value is not None:
                entry["api_key"] = value
                settings.save()
                print("Key saved.")
        elif choice == "m":
            value = prompt(
                "Model name (blank = default '{}'): ".format(registry["default_model"])
            ).strip()
            entry["model"] = value
            settings.save()
            print("Model set to {}.".format(value or "the default"))
        elif choice == "t":
            test_api_connection(model_id, settings)
        else:
            print("Please choose K, M, or T.")


def browser_login_now(site_id, settings):
    if not llm_browser.PLAYWRIGHT_AVAILABLE:
        print("Browser automation is not set up yet - use 'Browser automation setup' first.")
        return
    from playwright.sync_api import sync_playwright

    driver = make_driver(site_id)
    with sync_playwright() as playwright:
        context = open_site_context(playwright, site_id)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            if interactive_login(driver, page, settings.browser_model(site_id)):
                settings.save()
                print("Logged in to {}. The login is remembered for future runs.".format(
                    driver.display_name
                ))
        finally:
            try:
                context.close()
            except Exception:
                pass


def browser_site_settings(site_id, settings):
    display = SITE_INFO[site_id]["display_name"]
    while True:
        entry = settings.browser_model(site_id)
        print("\n{} settings:".format(display))
        print("  Username: {}   Password: {}   Last login OK: {}".format(
            entry.get("username") or "not set",
            "stored" if entry.get("password") else "not stored",
            (entry.get("last_login_ok") or "never")[:10],
        ))
        choice = prompt(
            "[U]sername  [P]assword  [L]og in now  or press Enter to go back: "
        ).strip().lower()
        if not choice:
            return
        if choice == "u":
            value = prompt("Username / email for {}: ".format(display)).strip()
            entry["username"] = value
            settings.save()
        elif choice == "p":
            print(
                "Leaving the password blank and typing it in the browser window "
                "yourself is recommended."
            )
            value = enter_secret("Password for {}".format(display))
            if value is not None:
                entry["password"] = value
                settings.save()
                print("Password saved.")
        elif choice == "l":
            browser_login_now(site_id, settings)
        else:
            print("Please choose U, P, or L.")


def playwright_setup():
    if llm_browser.PLAYWRIGHT_AVAILABLE:
        print("Browser automation (Playwright) is already installed.")
        return
    print(
        "\nThe browser models need one extra piece of software (Playwright) and\n"
        "a browser for it to drive. This is a one-time download of a few hundred MB."
    )
    if prompt("Install it now? [y/N]: ").strip().lower() not in ("y", "yes"):
        return
    for args, label in (
        ([sys.executable, "-m", "pip", "install", "playwright"], "Installing Playwright"),
        ([sys.executable, "-m", "playwright", "install", "chromium"], "Downloading the browser"),
    ):
        print("{}...".format(label))
        try:
            completed = subprocess.run(args)
        except OSError as e:
            print("Could not run the installer: {}".format(e))
            return
        if completed.returncode != 0:
            print(
                "That step did not finish successfully. Please try again, or ask "
                "for help with the message above."
            )
            return
    print(
        "Done. Please close this program and start it again so the browser "
        "models become available."
    )


def options_menu(settings):
    labels = [
        ("request_timeout_s", "Seconds to wait for an API answer"),
        ("max_retries", "How many times to retry a busy API"),
        ("browser_question_delay_s", "Pause between browser questions (seconds)"),
        ("answer_stable_seconds", "Seconds a browser answer must hold still to count as finished"),
        ("answer_max_wait_seconds", "Longest wait for one browser answer (seconds)"),
    ]
    while True:
        print("\nOptions:")
        for i, (key, label) in enumerate(labels, start=1):
            print("  {}. {}: {}".format(i, label, settings.option(key)))
        print("  {}. Fake test model for trying things out: {}".format(
            len(labels) + 1, "ON" if settings.option("enable_test_model") else "off"
        ))
        raw = prompt("Enter a number to change it, or press Enter to go back: ").strip()
        if not raw:
            return
        if raw == str(len(labels) + 1):
            settings.data["options"]["enable_test_model"] = not settings.option("enable_test_model")
            settings.save()
            continue
        if not raw.isdigit() or not 1 <= int(raw) <= len(labels):
            continue
        key, label = labels[int(raw) - 1]
        value = prompt("{} (currently {}): ".format(label, settings.option(key))).strip()
        if value.isdigit() and int(value) > 0:
            settings.data["options"][key] = int(value)
            settings.save()
        else:
            print("Please enter a positive whole number.")


def settings_menu(settings):
    while True:
        print(
            "\nSettings (stored in settings.json in this folder - keep that file private):"
        )
        for i, model_id in enumerate(API_MODEL_IDS, start=1):
            entry = settings.api_model(model_id)
            print("  {}. {:18s} key: {:10s} model: {}".format(
                i,
                API_REGISTRY[model_id]["display_name"],
                masked(entry.get("api_key")),
                entry.get("model") or "(default)",
            ))
        base = len(API_MODEL_IDS)
        for j, site_id in enumerate(BROWSER_MODEL_IDS, start=1):
            entry = settings.browser_model(site_id)
            print("  {}. {:18s} username: {:20s} login: {}".format(
                base + j,
                SITE_INFO[site_id]["display_name"],
                (entry.get("username") or "not set")[:20],
                "verified " + entry["last_login_ok"][:10] if entry.get("last_login_ok") else "never",
            ))
        setup_item = base + len(BROWSER_MODEL_IDS) + 1
        options_item = setup_item + 1
        print("  {}. Browser automation setup (Playwright: {})".format(
            setup_item, "installed" if llm_browser.PLAYWRIGHT_AVAILABLE else "NOT INSTALLED"
        ))
        print("  {}. Options (timeouts, retries, browser pacing, test model)".format(options_item))
        raw = prompt("Enter a number, or press Enter to go back: ").strip()
        if not raw:
            return
        if not raw.isdigit():
            continue
        number = int(raw)
        if 1 <= number <= base:
            api_model_settings(API_MODEL_IDS[number - 1], settings)
        elif base < number <= base + len(BROWSER_MODEL_IDS):
            browser_site_settings(BROWSER_MODEL_IDS[number - base - 1], settings)
        elif number == setup_item:
            playwright_setup()
        elif number == options_item:
            options_menu(settings)


# ---------- main ----------


def main():
    print("=" * 60)
    print("LLM Medical Cases - LLM Runner (for the PI)")
    print("=" * 60)
    try:
        settings = SettingsStore.load_or_create()
        case_sets = CaseSetStore.load_or_create()
        answers = AnswersStore.load_or_create()
        master = MasterStore.load(MASTER_FILENAME) if os.path.exists(MASTER_FILENAME) else None
    except (CaseStoreError, OSError) as e:
        print("Error: {}".format(e))
        prompt("Press Enter to close. ")
        return 1

    if answers.prompt_template_version is None:
        answers.prompt_template_version = PROMPT_TEMPLATE_VERSION
    elif answers.prompt_template_version != PROMPT_TEMPLATE_VERSION:
        print(
            "Note: earlier answers were collected with a different wording of the\n"
            "question sent to the models (version {} vs {} now).".format(
                answers.prompt_template_version, PROMPT_TEMPLATE_VERSION
            )
        )

    if master is None:
        print(
            "\nNo master case database ({}) was found in this folder.\n"
            "Run merge_cases.py first, or copy the file here. You can still "
            "open Settings.".format(MASTER_FILENAME)
        )
    else:
        providers = {c["provider_number"] for c in master.cases.values()}
        print("Master file: {} - {} cases from {} provider{}.".format(
            MASTER_FILENAME, len(master.cases), len(providers),
            "" if len(providers) == 1 else "s",
        ))
        print("Answers so far: {}.".format(len(answers.answers)))

    while True:
        if master is None:
            choice = prompt("\n[S]ettings  [Q]uit > ").strip().lower()
        else:
            choice = prompt(
                "\n[R]un cases through LLMs  [C]ase sets  [A]nswers so far  "
                "[S]ettings  [Q]uit > "
            ).strip().lower()
        try:
            if choice == "q":
                print("All answers are saved in answers.json.")
                prompt("Press Enter to close. ")
                return 0
            if choice == "s":
                settings_menu(settings)
            elif master is not None and choice == "r":
                run_flow(master, answers, settings, case_sets)
            elif master is not None and choice == "c":
                case_sets_menu(master, case_sets)
            elif master is not None and choice == "a":
                answers_menu(master, answers, settings)
            elif choice:
                print("Please choose one of the letters shown.")
        except CaseStoreError as e:
            print("Error: {}".format(e))
        except KeyboardInterrupt:
            print("\n(Interrupted - back to the main menu. Everything is saved.)")


if __name__ == "__main__":
    raise SystemExit(main())
