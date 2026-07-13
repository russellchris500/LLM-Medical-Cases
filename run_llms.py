#!/usr/bin/env python3
"""Program 3 of the LLM Medical Cases evaluation framework: the LLM Runner.

The principal investigator (PI) uses this tool to run cases from the master
database (master_cases.json, built by Program 2) through a chosen set of
LLMs, and to store every answer - text and images - in answers.json and
answer_images/ for later scoring.

It is a window-based program - run it (or double-click "Run LLMs.pyw")
and work in the window.

Models:
- API models (Anthropic Claude, OpenAI GPT, Google Gemini, xAI Grok) run
  unattended once an API key is entered in Settings.
- Browser models (OpenEvidence, UpToDate, Doximity GPT) are driven through
  a visible browser window; the PI should stay at the computer for that
  part in case a site asks for a login or verification.

Requires Python 3.8+. The API models need nothing installed; the browser
models need Playwright, which the Settings menu can install for you.
"""

import os
import subprocess
import sys
import time
import random

from case_editor import CaseStoreError, now_iso
from merge_cases import MASTER_FILENAME, MasterStore
from eval_common import (
    AnswersStore,
    CaseSetStore,
    OK_STATUSES,
    SelectionError,
    SettingsStore,
    case_hash,
    model_slug,
    parse_selection,
    sort_case_ids,
    split_case_id,
)
from llm_api import API_MODEL_IDS, API_REGISTRY, ApiCallError, ModelAbort, call_api_model, resolve_model
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
    """Every model the runner knows, in menu order.

    Each entry carries the MODEL NAME that, together with the LLM, forms
    the unique scored identity (variant_id): the same LLM running two
    different model names is scored and ranked as two separate models.
    API models resolve their name from Settings (or the built-in default);
    browser sites have no API to ask, so the user types the model name in
    Settings and must do so before the site can be run.
    """
    catalog = []
    for model_id in API_MODEL_IDS:
        model_name = resolve_model(model_id, settings.api_model(model_id))
        catalog.append(
            {
                "model_id": model_id,
                "display_name": API_REGISTRY[model_id]["display_name"],
                "kind": "api",
                "model_name": model_name,
            }
        )
    for site_id in BROWSER_MODEL_IDS:
        model_name = (settings.browser_model(site_id).get("model") or "").strip()
        catalog.append(
            {
                "model_id": site_id,
                "display_name": SITE_INFO[site_id]["display_name"],
                "kind": "browser",
                "model_name": model_name,
            }
        )
    if settings.option("enable_test_model"):
        catalog.append(
            {
                "model_id": TEST_MODEL_ID,
                "display_name": "Test model (fake)",
                "kind": "test",
                "model_name": "test-model-1",
            }
        )
    for entry in catalog:
        entry["variant_id"] = model_slug(entry["model_id"], entry["model_name"])
        entry["scored_as"] = (
            "{} ({})".format(entry["display_name"], entry["model_name"])
            if entry["model_name"] else entry["display_name"]
        )
    return catalog


def new_record(case, model, prompt_sent, deep_thinking=True):
    return {
        "case_id": case["case_id"],
        # model_id is the full scored identity: LLM + model name.
        "model_id": model["variant_id"],
        "llm_id": model["model_id"],
        "model_name": model["model_name"],
        "model_kind": model["kind"],
        "model_display_name": model["scored_as"],
        "deep_thinking": deep_thinking,
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


# ---------- worklist ----------


def build_worklist(master, answers, case_ids, models):
    """Split the requested (case, model) pairs by what still needs asking."""
    todo = {model["variant_id"]: [] for model in models}
    skipped = 0
    failed_pairs = []
    changed_pairs = []
    for model in models:
        for case_id in case_ids:
            existing = answers.get(case_id, model["variant_id"])
            if existing is None:
                todo[model["variant_id"]].append(case_id)
            elif existing.get("status") in OK_STATUSES:
                if existing.get("case_sha256") != case_hash(master.cases[case_id]):
                    changed_pairs.append((case_id, model))
                else:
                    skipped += 1
            else:
                failed_pairs.append((case_id, model))
    return todo, skipped, failed_pairs, changed_pairs


# ---------- running (UI-agnostic: works with any ui offering log/ask/tell) ----------


class AbandonRun(Exception):
    """The user pressed Stop."""


def run_test_model(case):
    return {
        "response_text": "TEST ANSWER for {}: this canned reply comes from the "
        "built-in fake model used to try the programs end to end.".format(case["case_id"]),
        "model_requested": "test-model-1",
        "model_reported": "test-model-1",
        "attempts": 1,
    }


def run_api_phase(master, answers, settings, models, todo, ui):
    api_models = [m for m in models if m["kind"] in ("api", "test") and todo[m["variant_id"]]]
    if not api_models:
        return
    total = sum(len(todo[m["variant_id"]]) for m in api_models)
    ui.log("API models ({} answers to collect):".format(total))
    done = 0
    options = settings.data["options"]
    for model in api_models:
        for case_id in todo[model["variant_id"]]:
            if ui.stop_requested:
                raise AbandonRun()
            done += 1
            case = master.cases[case_id]
            record = new_record(
                case, model, render_prompt(case),
                deep_thinking=bool(settings.option("deep_thinking")),
            )
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
                        log=ui.log,
                    )
                record.update(result)
                record["status"] = "ok"
                record["finished_at"] = now_iso()
                answers.upsert(record)
                ui.log("  [{}/{}] {} x {} - ok ({:.1f}s)".format(
                    done, total, case_id, model["scored_as"],
                    time.monotonic() - started,
                ))
            except ApiCallError as error:
                record["error"] = str(error)
                record["finished_at"] = now_iso()
                answers.upsert(record)
                ui.log("  [{}/{}] {} x {} - FAILED: {}".format(
                    done, total, case_id, model["display_name"], error
                ))
            except ModelAbort as error:
                ui.log("  {} is being skipped for the rest of this run: {}".format(
                    model["display_name"], error
                ))
                break


def interactive_login(driver, page, site_settings, ui):
    try:
        page.goto(driver.login_url, wait_until="domcontentloaded")
    except Exception:
        pass
    driver.wait_until_ready(page)
    driver.autofill_login(
        page, site_settings.get("username", ""), site_settings.get("password", "")
    )
    while True:
        choice = ui.ask_choice(
            "Log in to {}".format(driver.display_name),
            "A browser window is open on {}. Please finish logging in there,\n"
            "including any verification code or \"I am not a robot\" check.\n"
            "If the site offers \"remember this device\", say yes.\n\n"
            "TIP: if signing in with Google complains that the browser is not\n"
            "safe, use the site's own email-and-password (or emailed code)\n"
            "sign-in instead of the \"Continue with Google\" button - you only\n"
            "need to do this once; the login is remembered afterwards.\n\n"
            "When you can see the normal question page, click Continue.".format(
                driver.display_name
            ),
            [("check", "Continue - I am logged in"), ("stop", "Stop / skip this site")],
        )
        if choice != "check":
            return False
        try:
            page.goto(driver.home_url, wait_until="domcontentloaded")
        except Exception:
            pass
        if driver.is_logged_in(page):
            site_settings["last_login_ok"] = now_iso()
            return True
        ui.log("  It doesn't look logged in yet - please finish in the browser window.")


def manual_capture(driver, page, prompt_text, images_dir, basename, ui):
    if copy_to_clipboard(page, prompt_text):
        message = (
            "The case text is on your clipboard.\n\nIn the browser window: paste "
            "it (Ctrl+V), send it, and wait for the FULL answer to appear.\n\n"
            "Then click OK here to capture it."
        )
    else:
        ui.log("Copy the case text between the lines below into the site:")
        ui.log("-" * 50)
        ui.log(prompt_text)
        ui.log("-" * 50)
        message = (
            "The case text is printed in the log pane. Copy it into the browser, "
            "send it, wait for the FULL answer, then click OK to capture it."
        )
    ui.tell("Your turn in the browser", message)
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


def run_browser_site(master, answers, settings, model, case_ids, ui):
    site_id = model["model_id"]
    options = settings.data["options"]
    choice = ui.ask_choice(
        "Next: {}".format(model["display_name"]),
        "{} case{} will be asked on {}.\n\nA browser window will open; please "
        "stay at the computer in case the site asks you to log in or "
        "verify.".format(len(case_ids), "" if len(case_ids) == 1 else "s",
                         model["display_name"]),
        [
            ("auto", "Start (automatic)"),
            ("manual", "Start - I will drive every case by hand"),
            ("skip", "Skip {} for now".format(model["display_name"])),
        ],
    )
    if choice in (None, "skip"):
        return
    all_manual = choice == "manual"

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        context = open_site_context(playwright, site_id)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            driver = make_driver(site_id)
            driver.start_new_question(page)
            if not driver.is_logged_in(page):
                if not interactive_login(driver, page, settings.browser_model(site_id), ui):
                    ui.log("  Skipping {} (not logged in).".format(model["display_name"]))
                    return
                settings.save()

            images_dir = answers.ensure_images_dir()
            for index, case_id in enumerate(case_ids, start=1):
                if ui.stop_requested:
                    raise AbandonRun()
                case = master.cases[case_id]
                prompt_text = render_prompt(case)
                basename = answers.image_basename(case_id, model["variant_id"])
                record = new_record(case, model, prompt_text, deep_thinking=True)
                started = time.monotonic()
                ui.log("  [{}/{}] {} x {}...".format(
                    index, len(case_ids), case_id, model["display_name"]
                ))
                answers.clear_images(case_id, model["variant_id"])
                result = None
                mode_manual = all_manual
                while result is None:
                    try:
                        if mode_manual:
                            result = manual_capture(
                                driver, page, prompt_text, images_dir, basename, ui
                            )
                        else:
                            result = browser_ask_one(
                                driver, page, case, prompt_text, images_dir,
                                basename, options,
                            )
                    except BrowserStepError as error:
                        options_list = [
                            ("retry", "Retry automatically"),
                            ("manual", "I will do it by hand"),
                            ("skip", "Skip this case"),
                            ("abandon", "Set {} aside".format(model["display_name"])),
                        ]
                        if error.step == "logged_out":
                            options_list.insert(0, ("retry", "I have logged back in - continue"))
                            options_list = options_list[:1] + options_list[2:]
                        answer = ui.ask_choice(
                            "Problem on {} with case {}".format(
                                model["display_name"], case_id
                            ),
                            "{}.\n\nThe browser window is still open.".format(error.detail),
                            options_list,
                        )
                        if answer == "retry":
                            continue
                        if answer == "manual":
                            mode_manual = True
                            continue
                        if answer in (None, "skip"):
                            record["error"] = "skipped: " + error.detail
                            record["finished_at"] = now_iso()
                            answers.upsert(record)
                            break
                        if answer == "abandon":
                            ui.log("  Set {} aside for this run.".format(model["display_name"]))
                            return
                if result is None:
                    continue
                record["response_text"] = result.text
                record["images"] = [p.replace(os.sep, "/") for p in result.image_paths]
                record["model_requested"] = model["model_name"] or model["display_name"]
                record["model_reported"] = result.model_reported
                record["status"] = "ok_manual" if result.manual else "ok"
                record["finished_at"] = now_iso()
                answers.upsert(record)
                ui.log("        ok{}, {} image{} ({:.0f}s)".format(
                    " (by hand)" if result.manual else "",
                    len(result.image_paths),
                    "" if len(result.image_paths) == 1 else "s",
                    time.monotonic() - started,
                ))
                if index < len(case_ids) and not mode_manual:
                    delay = options.get("browser_question_delay_s", 8)
                    time.sleep(delay * random.uniform(0.5, 1.5))
        finally:
            try:
                context.close()
            except Exception:
                pass


def run_everything(master, answers, settings, models, todo, ui):
    try:
        run_api_phase(master, answers, settings, models, todo, ui)
        browser_models = [m for m in models if m["kind"] == "browser" and todo[m["variant_id"]]]
        for model in browser_models:
            if ui.stop_requested:
                raise AbandonRun()
            run_browser_site(master, answers, settings, model, todo[model["variant_id"]], ui)
    except AbandonRun:
        ui.log("Stopped. Everything answered so far is saved; run again to continue.")
    summarize_run(answers, sorted({c for ids in todo.values() for c in ids}), models, ui)


def summarize_run(answers, case_ids, models, ui):
    ok = manual = failed = missing = 0
    failures = []
    for model in models:
        for case_id in case_ids:
            record = answers.get(case_id, model["variant_id"])
            if record is None:
                missing += 1
            elif record["status"] == "ok":
                ok += 1
            elif record["status"] == "ok_manual":
                manual += 1
            else:
                failed += 1
                failures.append("{} x {}: {}".format(
                    case_id, model["display_name"], record.get("error")
                ))
    line = "Done. {} ok".format(ok)
    if manual:
        line += ", {} ok (by hand)".format(manual)
    if failed:
        line += ", {} failed".format(failed)
    if missing:
        line += ", {} not attempted".format(missing)
    ui.log(line + ".")
    for failure in failures[:10]:
        ui.log("  " + failure)
    ui.log("Everything is saved after each answer; finished pairs are skipped next time.")
# ---------- window interface ----------


def masked(secret):
    secret = (secret or "").strip()
    return "not set" if not secret else "..." + secret[-4:]


class RunnerApp:
    def __init__(self, root, master, answers, settings, case_sets, gui):
        self.root = root
        self.master = master
        self.answers = answers
        self.settings = settings
        self.case_sets = case_sets
        self.gui = gui
        self.selected_cases = []
        self.model_vars = {}
        self.current_ui = None
        import gui_common

        self.gc = gui_common
        tk = gui.tk
        self.notebook = gui.ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.run_tab = tk.Frame(self.notebook)
        self.sets_tab = tk.Frame(self.notebook)
        self.answers_tab = tk.Frame(self.notebook)
        self.settings_tab = tk.Frame(self.notebook)
        self.notebook.add(self.run_tab, text="  Run  ")
        self.notebook.add(self.sets_tab, text="  Case sets  ")
        self.notebook.add(self.answers_tab, text="  Answers so far  ")
        self.notebook.add(self.settings_tab, text="  Settings  ")
        self._build_run_tab()
        self._build_sets_tab()
        self._build_answers_tab()
        self._build_settings_tab()
        self.task = gui_common.BackgroundTask(root, self.run_log.log)
        self.settings_task = gui_common.BackgroundTask(root, self.settings_log.log)
        self.refresh_all()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- Run tab ----------------

    def _build_run_tab(self):
        tk = self.gui.tk
        frame = self.run_tab
        self.master_label = tk.Label(frame, text="", anchor="w")
        self.master_label.pack(fill="x", pady=(6, 0))

        picker = tk.LabelFrame(frame, text="1. Which cases?")
        picker.pack(fill="x", pady=6)
        self.case_mode = tk.StringVar(value="all")
        row1 = tk.Frame(picker)
        row1.pack(fill="x", padx=6, pady=2)
        tk.Radiobutton(row1, text="All cases", variable=self.case_mode, value="all",
                       command=self.resolve_cases).pack(side="left")
        row2 = tk.Frame(picker)
        row2.pack(fill="x", padx=6, pady=2)
        tk.Radiobutton(row2, text="These cases:", variable=self.case_mode, value="expr",
                       command=self.resolve_cases).pack(side="left")
        self.case_expression = tk.Entry(row2, width=44)
        self.case_expression.pack(side="left", padx=4)
        tk.Label(row2, text="(e.g. 003-001..003-020, 005-004, provider 7)").pack(side="left")
        row3 = tk.Frame(picker)
        row3.pack(fill="x", padx=6, pady=2)
        tk.Radiobutton(row3, text="Saved case set:", variable=self.case_mode, value="set",
                       command=self.resolve_cases).pack(side="left")
        self.set_choice = self.gui.ttk.Combobox(row3, width=30, state="readonly")
        self.set_choice.pack(side="left", padx=4)
        self.set_choice.bind("<<ComboboxSelected>>", lambda e: self.resolve_cases())
        row4 = tk.Frame(picker)
        row4.pack(fill="x", padx=6, pady=(2, 6))
        tk.Button(row4, text="Check selection", command=self.resolve_cases).pack(side="left")
        tk.Button(row4, text="Save selection as a named set",
                  command=self.save_selection_as_set).pack(side="left", padx=6)
        self.selection_label = tk.Label(row4, text="", anchor="w")
        self.selection_label.pack(side="left", fill="x", expand=True, padx=6)

        models_frame = tk.LabelFrame(frame, text="2. Which LLMs?")
        models_frame.pack(fill="x", pady=6)
        self.models_holder = tk.Frame(models_frame)
        self.models_holder.pack(fill="x", padx=6, pady=4)

        options_row = tk.Frame(frame)
        options_row.pack(fill="x", pady=(0, 4))
        self.retry_failed = tk.BooleanVar(value=True)
        self.reask_changed = tk.BooleanVar(value=False)
        tk.Checkbutton(options_row, text="Retry pairs that failed before",
                       variable=self.retry_failed).pack(side="left")
        tk.Checkbutton(options_row, text="Re-ask answers whose case wording changed",
                       variable=self.reask_changed).pack(side="left", padx=10)
        self.start_button = tk.Button(options_row, text="3. Start the run",
                                      command=self.start_run)
        self.start_button.pack(side="right")
        self.stop_button = tk.Button(options_row, text="Stop after the current answer",
                                     command=self.stop_run, state="disabled")
        self.stop_button.pack(side="right", padx=6)

        self.run_log = self.gc.LogBox(frame, height=14).pack(fill="both", expand=True, pady=4)

    def refresh_all(self):
        self.refresh_master_label()
        self.refresh_set_choices()
        self.refresh_model_checkboxes()
        self.resolve_cases()
        self.refresh_sets_tab()
        self.refresh_answers_tab()
        self.refresh_settings_rows()

    def refresh_master_label(self):
        if self.master is None:
            self.master_label.configure(
                text="No master case database (master_cases.json) was found in this "
                "folder - run the Case Merger first. Settings still work."
            )
            self.start_button.configure(state="disabled")
        else:
            providers = {c["provider_number"] for c in self.master.cases.values()}
            self.master_label.configure(
                text="Master database: {} cases from {} provider{}.  Answers so far: {}.".format(
                    len(self.master.cases), len(providers),
                    "" if len(providers) == 1 else "s", len(self.answers.answers),
                )
            )

    def refresh_set_choices(self):
        names = sorted(self.case_sets.sets, key=str.lower)
        self.set_choice.configure(values=names)

    def refresh_model_checkboxes(self):
        # Keep whatever is already ticked - refreshing the status text must
        # never silently clear the user's model selection.
        previous = {m: var.get() for m, (var, _model) in self.model_vars.items()}
        for child in self.models_holder.winfo_children():
            child.destroy()
        self.model_vars = {}
        tk = self.gui.tk
        for model in model_catalog(self.settings):
            var = tk.BooleanVar(value=previous.get(model["model_id"], False))
            self.model_vars[model["model_id"]] = (var, model)
            status = self.model_status(model)
            tk.Checkbutton(
                self.models_holder,
                text="{}   ({})".format(model["display_name"], status),
                variable=var, anchor="w",
            ).pack(fill="x")

    def model_status(self, model):
        if model["kind"] == "api":
            key = self.settings.api_model(model["model_id"]).get("api_key", "")
            status = "API, key set" if key.strip() else "API, NO KEY - set it in Settings"
        elif model["kind"] == "browser":
            if not model["model_name"]:
                status = "browser, MODEL NAME NEEDED - set it in Settings"
            elif not llm_browser.PLAYWRIGHT_AVAILABLE:
                status = "browser, needs one-time setup in Settings"
            else:
                last = self.settings.browser_model(model["model_id"]).get("last_login_ok")
                status = "browser, login OK {}".format(last[:10]) if last else "browser, never logged in"
        else:
            status = "fake test model"
        if self.selected_cases:
            answered = sum(
                1 for c in self.selected_cases
                if (self.answers.get(c, model["variant_id"]) or {}).get("status") in OK_STATUSES
            )
            status += "; answered {}/{}".format(answered, len(self.selected_cases))
        return status

    def resolve_cases(self):
        if self.master is None:
            return
        mode = self.case_mode.get()
        warnings = []
        if mode == "all":
            selected = sort_case_ids(self.master.cases)
        elif mode == "set":
            name = self.set_choice.get()
            if not name:
                self.selection_label.configure(text="Pick a saved set from the list.")
                self.selected_cases = []
                return
            selected, missing = self.case_sets.resolve(name, self.master.cases)
            if missing:
                warnings.append("{} case(s) in the set are no longer in the master".format(
                    len(missing)
                ))
        else:
            expression = self.case_expression.get().strip()
            if not expression:
                self.selection_label.configure(text="Type the cases first.")
                self.selected_cases = []
                return
            try:
                selected, notes = parse_selection(expression, self.master.cases)
            except SelectionError as error:
                self.selection_label.configure(text=str(error))
                self.selected_cases = []
                return
            warnings.extend(notes)
        self.selected_cases = selected
        text = "{} case{} selected.".format(len(selected), "" if len(selected) == 1 else "s")
        if warnings:
            text += "  (" + "; ".join(warnings) + ")"
        self.selection_label.configure(text=text)
        self.refresh_model_checkboxes()

    def save_selection_as_set(self):
        self.resolve_cases()
        if not self.selected_cases:
            return
        name = self.gc.ask_string(self.root, "Save case set", "Name for this set:")
        if not name:
            return
        expression = (
            self.case_expression.get().strip()
            if self.case_mode.get() == "expr" else "chosen in the window"
        )
        try:
            self.case_sets.add(name, expression, self.selected_cases)
        except CaseStoreError as error:
            self.gui.messagebox.showerror("Not saved", str(error), parent=self.root)
            return
        self.refresh_set_choices()
        self.refresh_sets_tab()
        self.run_log.log("Saved case set '{}' ({} cases).".format(name, len(self.selected_cases)))

    def chosen_models(self):
        chosen = []
        for model_id, (var, model) in self.model_vars.items():
            if not var.get():
                continue
            if model["kind"] == "api" and not self.settings.api_model(model_id).get(
                "api_key", ""
            ).strip():
                self.run_log.log(
                    "Skipping {}: no API key (see Settings).".format(model["display_name"])
                )
            elif model["kind"] == "browser" and not model["model_name"]:
                self.run_log.log(
                    "Skipping {}: set its model name in Settings first (which "
                    "model the site runs, e.g. 'GPT-5' - the LLM plus the model "
                    "name is what gets scored).".format(model["display_name"])
                )
            elif model["kind"] == "browser" and not llm_browser.PLAYWRIGHT_AVAILABLE:
                self.run_log.log(
                    "Skipping {}: browser automation is not set up yet (see "
                    "Settings).".format(model["display_name"])
                )
            else:
                chosen.append(model)
        return chosen

    def start_run(self):
        if self.task.running or self.settings_task.running:
            self.gui.messagebox.showinfo(
                "Already busy", "Please wait for the current job to finish.",
                parent=self.root,
            )
            return
        self.resolve_cases()
        if not self.selected_cases:
            self.gui.messagebox.showinfo(
                "No cases", "Choose the cases first (step 1).", parent=self.root
            )
            return
        models = self.chosen_models()
        if not models:
            self.gui.messagebox.showinfo(
                "No LLMs", "Tick at least one ready LLM (step 2).", parent=self.root
            )
            return
        todo, skipped, failed_pairs, changed_pairs = build_worklist(
            self.master, self.answers, self.selected_cases, models
        )
        if failed_pairs and self.retry_failed.get():
            for case_id, model in failed_pairs:
                todo[model["variant_id"]].append(case_id)
        if changed_pairs and self.reask_changed.get():
            for case_id, model in changed_pairs:
                todo[model["variant_id"]].append(case_id)
        for model_id in todo:
            todo[model_id] = sort_case_ids(set(todo[model_id]))
        api_count = sum(len(todo[m["variant_id"]]) for m in models if m["kind"] in ("api", "test"))
        browser_models = [m for m in models if m["kind"] == "browser" and todo[m["variant_id"]]]
        browser_count = sum(len(todo[m["variant_id"]]) for m in browser_models)
        if api_count + browser_count == 0:
            self.gui.messagebox.showinfo(
                "Nothing to do",
                "Everything selected is already answered ({} pairs skipped).".format(skipped),
                parent=self.root,
            )
            return
        message = "{} answers to collect:\n- {} by API (unattended)".format(
            api_count + browser_count, api_count
        )
        if browser_models:
            message += "\n- {} in the browser via {} (please stay at the computer)".format(
                browser_count, ", ".join(m["display_name"] for m in browser_models)
            )
        if skipped:
            message += "\n\n{} already-answered pairs will be skipped.".format(skipped)
        if changed_pairs and not self.reask_changed.get():
            message += "\n\nNote: {} answered pair(s) have changed case wording (not re-asked).".format(
                len(changed_pairs)
            )
        if not self.gui.messagebox.askyesno("Start the run?", message, parent=self.root):
            return
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.run_log.log("=" * 50)

        def work(ui):
            run_everything(self.master, self.answers, self.settings, models, todo, ui)

        def done(_result, error):
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            if error is not None:
                self.run_log.log("The run stopped with a problem: {}".format(error))
            self.refresh_answers_tab()
            self.refresh_model_checkboxes()
            self.refresh_master_label()

        self.current_ui = self.task.start(work, on_done=done)

    def stop_run(self):
        if self.current_ui is not None:
            self.current_ui.stop_requested = True
            self.run_log.log("Stopping after the current answer...")

    # ---------------- Case sets tab ----------------

    def _build_sets_tab(self):
        tk = self.gui.tk
        frame = self.sets_tab
        tk.Label(
            frame, text="Saved case selections. Save one on the Run tab, then reuse "
            "it anytime (for example to run the same cases on a new LLM).",
            anchor="w", justify="left",
        ).pack(fill="x", pady=(6, 4))
        body = tk.Frame(frame)
        body.pack(fill="both", expand=True)
        self.sets_list = tk.Listbox(body, width=34, exportselection=False)
        self.sets_list.pack(side="left", fill="y", pady=4)
        self.sets_list.bind("<<ListboxSelect>>", lambda e: self.show_set_details())
        self.set_details = tk.Label(body, text="", anchor="nw", justify="left")
        self.set_details.pack(side="left", fill="both", expand=True, padx=10, pady=4)
        row = tk.Frame(frame)
        row.pack(fill="x", pady=4)
        tk.Button(row, text="Rename", command=self.rename_set).pack(side="left")
        tk.Button(row, text="Delete", command=self.delete_set).pack(side="left", padx=6)

    def refresh_sets_tab(self):
        self.sets_list.delete(0, "end")
        for name in sorted(self.case_sets.sets, key=str.lower):
            entry = self.case_sets.sets[name]
            self.sets_list.insert("end", "{}  ({} cases)".format(name, len(entry["case_ids"])))
        self.set_details.configure(text="")

    def selected_set_name(self):
        selection = self.sets_list.curselection()
        if not selection:
            return None
        return sorted(self.case_sets.sets, key=str.lower)[selection[0]]

    def show_set_details(self):
        name = self.selected_set_name()
        if name is None:
            return
        entry = self.case_sets.sets[name]
        lines = ["Set '{}'".format(name), "Defined as: {}".format(entry["expression"]),
                 "Cases ({}):".format(len(entry["case_ids"]))]
        ids = entry["case_ids"]
        for i in range(0, len(ids), 8):
            lines.append("  " + ", ".join(ids[i:i + 8]))
        if self.master is not None:
            _present, missing = self.case_sets.resolve(name, self.master.cases)
            if missing:
                lines.append("No longer in the master: " + ", ".join(missing))
        self.set_details.configure(text="\n".join(lines[:30]))

    def rename_set(self):
        name = self.selected_set_name()
        if name is None:
            return
        new_name = self.gc.ask_string(self.root, "Rename set", "New name for '{}':".format(name))
        if not new_name:
            return
        try:
            self.case_sets.rename(name, new_name)
        except CaseStoreError as error:
            self.gui.messagebox.showerror("Not renamed", str(error), parent=self.root)
            return
        self.refresh_sets_tab()
        self.refresh_set_choices()

    def delete_set(self):
        name = self.selected_set_name()
        if name is None:
            return
        if self.gui.messagebox.askyesno(
            "Delete set", "Really delete the case set '{}'?".format(name), parent=self.root
        ):
            self.case_sets.delete(name)
            self.refresh_sets_tab()
            self.refresh_set_choices()

    # ---------------- Answers tab ----------------

    def _build_answers_tab(self):
        tk = self.gui.tk
        frame = self.answers_tab
        row = tk.Frame(frame)
        row.pack(fill="x", pady=(6, 0))
        tk.Button(row, text="Refresh", command=self.refresh_answers_tab).pack(side="left")
        tk.Button(row, text="View the selected answer", command=self.view_answer).pack(
            side="left", padx=6
        )
        self.answers_tree = self.gc.make_table(
            frame,
            [("case", "Case"), ("model", "LLM"), ("status", "Status"),
             ("images", "Images"), ("when", "Asked"), ("error", "Problem")],
            widths={"case": 80, "model": 150, "status": 90, "images": 60,
                    "when": 130, "error": 320},
        )
        self.answers_tree.master.pack(fill="both", expand=True, pady=6)

    def refresh_answers_tab(self):
        names = {m["variant_id"]: m["scored_as"] for m in model_catalog(self.settings)}
        tree = self.answers_tree
        for item in tree.get_children():
            tree.delete(item)
        ordered = sorted(
            self.answers.answers.items(),
            key=lambda kv: (split_case_id(kv[0][0]), kv[0][1]),
        )
        for (case_id, model_id), record in ordered:
            tree.insert("", "end", iid="{}|{}".format(case_id, model_id), values=(
                case_id,
                record.get("model_display_name") or names.get(model_id, model_id),
                record.get("status"),
                len(record.get("images", [])),
                (record.get("started_at") or "")[:16].replace("T", " "),
                (record.get("error") or "")[:80],
            ))

    def view_answer(self):
        selection = self.answers_tree.selection()
        if not selection:
            self.gui.messagebox.showinfo(
                "Nothing selected", "Click an answer in the table first.", parent=self.root
            )
            return
        case_id, model_id = selection[0].split("|", 1)
        record = self.answers.get(case_id, model_id)
        if record is None:
            return
        tk = self.gui.tk
        window = tk.Toplevel(self.root)
        window.title("{} x {}".format(case_id, record.get("model_display_name", model_id)))
        window.geometry("760x560")
        info = "Status: {}   Model: {}   Asked: {}".format(
            record.get("status"),
            record.get("model_reported") or record.get("model_requested"),
            record.get("started_at"),
        )
        tk.Label(window, text=info, anchor="w").pack(fill="x", padx=8, pady=(8, 0))
        text = self.gc.readonly_text(window, height=24)
        text.pack(fill="both", expand=True, padx=8, pady=8)
        body = record.get("response_text") or "(no text; error: {})".format(record.get("error"))
        if record.get("images"):
            body += "\n\nImages:\n" + "\n".join(record["images"])
        self.gc.set_text(text, body)

    # ---------------- Settings tab ----------------

    def _build_settings_tab(self):
        tk = self.gui.tk
        frame = self.settings_tab
        tk.Label(
            frame,
            text="Keys and passwords are stored in settings.json in this folder - "
            "keep that file private.",
            anchor="w",
        ).pack(fill="x", pady=(6, 2))
        self.settings_rows = tk.Frame(frame)
        self.settings_rows.pack(fill="x")
        self.settings_log = self.gc.LogBox(frame, height=8).pack(
            fill="both", expand=True, pady=6
        )

    def refresh_settings_rows(self):
        for child in self.settings_rows.winfo_children():
            child.destroy()
        tk = self.gui.tk
        holder = self.settings_rows
        row_index = 0
        tk.Label(holder, text="API models:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row_index, column=0, sticky="w", pady=(4, 0)
        )
        row_index += 1
        for model_id in API_MODEL_IDS:
            entry = self.settings.api_model(model_id)
            registry = API_REGISTRY[model_id]
            tk.Label(holder, text=registry["display_name"], width=18, anchor="w").grid(
                row=row_index, column=0, sticky="w"
            )
            tk.Label(holder, text="key: " + masked(entry.get("api_key")), width=14,
                     anchor="w").grid(row=row_index, column=1, sticky="w")
            tk.Button(
                holder, text="Set key",
                command=lambda m=model_id: self.set_api_key(m),
            ).grid(row=row_index, column=2, padx=2)
            tk.Label(
                holder,
                text="model: " + (entry.get("model") or "(default: {})".format(
                    registry["default_model"]
                )),
                anchor="w", width=30,
            ).grid(row=row_index, column=3, sticky="w")
            tk.Button(
                holder, text="Change model",
                command=lambda m=model_id: self.set_api_model_name(m),
            ).grid(row=row_index, column=4, padx=2)
            tk.Button(
                holder, text="Test",
                command=lambda m=model_id: self.test_api(m),
            ).grid(row=row_index, column=5, padx=2)
            row_index += 1
        tk.Label(holder, text="Browser sites:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row_index, column=0, sticky="w", pady=(8, 0)
        )
        row_index += 1
        for site_id in BROWSER_MODEL_IDS:
            entry = self.settings.browser_model(site_id)
            tk.Label(holder, text=SITE_INFO[site_id]["display_name"], width=18,
                     anchor="w").grid(row=row_index, column=0, sticky="w")
            model_name = (entry.get("model") or "").strip()
            tk.Label(
                holder,
                text="model: " + (model_name or "NOT SET"),
                width=24, anchor="w",
                fg="black" if model_name else "red",
            ).grid(row=row_index, column=1, sticky="w")
            tk.Button(
                holder, text="Set model name",
                command=lambda s=site_id: self.set_site_model_name(s),
            ).grid(row=row_index, column=2, padx=2)
            tk.Label(
                holder,
                text="user: {}; login: {}".format(
                    (entry.get("username") or "not set")[:16],
                    "verified " + entry["last_login_ok"][:10]
                    if entry.get("last_login_ok") else "never",
                ),
                width=34, anchor="w",
            ).grid(row=row_index, column=3, sticky="w")
            tk.Button(
                holder, text="Set login details",
                command=lambda s=site_id: self.set_site_login(s),
            ).grid(row=row_index, column=4, sticky="w", padx=2)
            tk.Button(
                holder, text="Log in now",
                command=lambda s=site_id: self.login_now(s),
            ).grid(row=row_index, column=5, padx=2)
            row_index += 1
        bottom = tk.Frame(holder)
        bottom.grid(row=row_index, column=0, columnspan=6, sticky="w", pady=(8, 0))
        tk.Button(
            bottom,
            text="Browser automation setup (Playwright: {})".format(
                "installed" if llm_browser.PLAYWRIGHT_AVAILABLE else "NOT INSTALLED"
            ),
            command=self.playwright_setup,
        ).pack(side="left")
        tk.Button(bottom, text="Options...", command=self.options_dialog).pack(
            side="left", padx=8
        )

    def set_api_key(self, model_id):
        value = self.gui.simpledialog.askstring(
            "API key",
            "Paste the API key for {} (it will be hidden):".format(
                API_REGISTRY[model_id]["display_name"]
            ),
            parent=self.root, show="*",
        )
        if value and value.strip():
            self.settings.api_model(model_id)["api_key"] = value.strip()
            self.settings.save()
            self.refresh_settings_rows()
            self.refresh_model_checkboxes()

    def set_api_model_name(self, model_id):
        registry = API_REGISTRY[model_id]
        value = self.gui.simpledialog.askstring(
            "Model name",
            "Model name for {} (leave empty for the default, {}).\n\nThe LLM "
            "plus this model name is what gets scored and ranked - changing "
            "it starts a separate scoring identity.".format(
                registry["display_name"], registry["default_model"]
            ),
            parent=self.root,
        )
        if value is None:
            return
        self.settings.api_model(model_id)["model"] = value.strip()
        self.settings.save()
        self.refresh_settings_rows()

    def test_api(self, model_id):
        if self.settings_task.running or self.task.running:
            return
        options = dict(self.settings.data["options"])
        options["max_retries"] = 1
        entry = self.settings.api_model(model_id)
        display = API_REGISTRY[model_id]["display_name"]
        self.settings_log.log("Contacting {}...".format(display))

        def work(ui):
            return call_api_model(
                model_id, entry, "Reply with the single word OK.", options, log=ui.log
            )

        def done(result, error):
            if error is not None:
                self.settings_log.log("Test failed: {}".format(error))
            else:
                self.settings_log.log(
                    "Success. {} answered and reported model '{}'.".format(
                        display, result["model_reported"]
                    )
                )

        self.settings_task.start(work, on_done=done)

    def set_site_model_name(self, site_id):
        display = SITE_INFO[site_id]["display_name"]
        entry = self.settings.browser_model(site_id)
        value = self.gui.simpledialog.askstring(
            "Model name",
            "Which model does {} run? This name, together with the LLM, is\n"
            "what gets scored and ranked (e.g. 'GPT-5', 'OpenEvidence "
            "2026-07').\nChanging it later starts a separate scoring "
            "identity.".format(display),
            parent=self.root, initialvalue=entry.get("model", ""),
        )
        if value is None:
            return
        entry["model"] = value.strip()
        self.settings.save()
        self.refresh_settings_rows()
        self.refresh_model_checkboxes()

    def set_site_login(self, site_id):
        display = SITE_INFO[site_id]["display_name"]
        entry = self.settings.browser_model(site_id)
        username = self.gui.simpledialog.askstring(
            "Username", "Username / email for {}:".format(display),
            parent=self.root, initialvalue=entry.get("username", ""),
        )
        if username is None:
            return
        entry["username"] = username.strip()
        password = self.gui.simpledialog.askstring(
            "Password (optional)",
            "Password for {} (hidden). Leaving this EMPTY and typing it in the "
            "browser window yourself is recommended:".format(display),
            parent=self.root, show="*",
        )
        if password:
            entry["password"] = password.strip()
        self.settings.save()
        self.refresh_settings_rows()

    def login_now(self, site_id):
        if not llm_browser.PLAYWRIGHT_AVAILABLE:
            self.gui.messagebox.showinfo(
                "Setup needed",
                "Browser automation is not set up yet - click 'Browser automation "
                "setup' first.",
                parent=self.root,
            )
            return
        if self.settings_task.running or self.task.running:
            return

        def work(ui):
            from playwright.sync_api import sync_playwright

            driver = make_driver(site_id)
            with sync_playwright() as playwright:
                context = open_site_context(playwright, site_id)
                try:
                    page = context.pages[0] if context.pages else context.new_page()
                    if interactive_login(
                        driver, page, self.settings.browser_model(site_id), ui
                    ):
                        self.settings.save()
                        ui.log("Logged in to {}. The login is remembered for future "
                               "runs.".format(driver.display_name))
                finally:
                    try:
                        context.close()
                    except Exception:
                        pass

        self.settings_task.start(work, on_done=lambda r, e: self.refresh_settings_rows())

    def playwright_setup(self):
        if llm_browser.PLAYWRIGHT_AVAILABLE:
            self.gui.messagebox.showinfo(
                "Already installed", "Browser automation (Playwright) is already installed.",
                parent=self.root,
            )
            return
        if not self.gui.messagebox.askyesno(
            "One-time setup",
            "The browser models need one extra piece of software (Playwright) and "
            "a browser for it to drive - a one-time download of a few hundred "
            "MB.\n\nInstall it now?",
            parent=self.root,
        ):
            return
        if self.settings_task.running or self.task.running:
            return

        def work(ui):
            for args, label in (
                ([sys.executable, "-m", "pip", "install", "playwright"],
                 "Installing Playwright"),
                ([sys.executable, "-m", "playwright", "install", "chromium"],
                 "Downloading the browser"),
            ):
                ui.log(label + "...")
                process = subprocess.Popen(
                    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in process.stdout:
                    ui.log("  " + line.rstrip())
                if process.wait() != 0:
                    ui.log("That step did not finish successfully - please try again "
                           "or ask for help with the messages above.")
                    return
            ui.log("Done. Please close this program and start it again so the "
                   "browser models become available.")

        self.settings_task.start(work)

    def options_dialog(self):
        tk = self.gui.tk
        dialog = tk.Toplevel(self.root)
        dialog.title("Options")
        dialog.transient(self.root)
        dialog.grab_set()
        labels = [
            ("request_timeout_s", "Seconds to wait for an API answer"),
            ("max_retries", "How many times to retry a busy API"),
            ("browser_question_delay_s", "Pause between browser questions (seconds)"),
            ("answer_stable_seconds", "Seconds a browser answer must hold still"),
            ("answer_max_wait_seconds", "Longest wait for one browser answer (seconds)"),
        ]
        entries = {}
        for i, (key, label) in enumerate(labels):
            tk.Label(dialog, text=label + ":", anchor="w").grid(
                row=i, column=0, sticky="w", padx=10, pady=3
            )
            entry = tk.Entry(dialog, width=8)
            entry.insert(0, str(self.settings.option(key)))
            entry.grid(row=i, column=1, padx=10)
            entries[key] = entry
        deep = tk.BooleanVar(value=bool(self.settings.option("deep_thinking")))
        tk.Checkbutton(
            dialog, text="Deep thinking (models reason at length before answering)",
            variable=deep, anchor="w",
        ).grid(row=len(labels), column=0, columnspan=2, sticky="w", padx=10, pady=3)
        test_model = tk.BooleanVar(value=bool(self.settings.option("enable_test_model")))
        tk.Checkbutton(
            dialog, text="Fake test model for trying things out",
            variable=test_model, anchor="w",
        ).grid(row=len(labels) + 1, column=0, columnspan=2, sticky="w", padx=10, pady=3)

        def save():
            for key, entry in entries.items():
                value = entry.get().strip()
                if value.isdigit() and int(value) > 0:
                    self.settings.data["options"][key] = int(value)
            self.settings.data["options"]["deep_thinking"] = deep.get()
            self.settings.data["options"]["enable_test_model"] = test_model.get()
            self.settings.save()
            dialog.destroy()
            self.refresh_model_checkboxes()

        tk.Button(dialog, text="Save", width=10, command=save).grid(
            row=len(labels) + 2, column=0, pady=10
        )
        tk.Button(dialog, text="Cancel", width=10, command=dialog.destroy).grid(
            row=len(labels) + 2, column=1, pady=10
        )

    def on_close(self):
        if self.task.running:
            if not self.gui.messagebox.askyesno(
                "A run is in progress",
                "A run is still working. Everything answered so far is saved.\n\n"
                "Close anyway?",
                parent=self.root,
            ):
                return
        self.root.destroy()


class _GuiModules:
    def __init__(self, tk, ttk, messagebox, simpledialog):
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.simpledialog = simpledialog


def main():
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog
    import gui_common

    root = gui_common.make_root("LLM Runner (for the PI)", 1020, 720)
    try:
        settings = SettingsStore.load_or_create()
        case_sets = CaseSetStore.load_or_create()
        answers = AnswersStore.load_or_create()
        master = MasterStore.load(MASTER_FILENAME) if os.path.exists(MASTER_FILENAME) else None
    except (CaseStoreError, OSError) as error:
        gui_common.show_error("Cannot start", str(error))
        root.destroy()
        return 1

    if answers.prompt_template_version is None:
        answers.prompt_template_version = PROMPT_TEMPLATE_VERSION
    app = RunnerApp(
        root, master, answers, settings, case_sets,
        _GuiModules(tk, ttk, messagebox, simpledialog),
    )
    if answers.answers and AnswersStore.load_or_create().prompt_template_version not in (
        None, PROMPT_TEMPLATE_VERSION
    ):
        app.run_log.log(
            "Note: earlier answers were collected with a different wording of the "
            "question sent to the models."
        )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
