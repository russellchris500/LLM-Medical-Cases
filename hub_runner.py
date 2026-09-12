"""Run AI Answers: runs the study website's runs on this computer.

Double-click "Run AI Answers.pyw". The program signs into the website
with the token from its "Connect my computer" page, lists the runs
waiting for you (your own, or - for the PI - runs graders sent over),
and executes them with the exact same machinery as the classic Runner:
API models in parallel, browser sites in a visible window with you
nearby. Every answer is uploaded the moment it is captured, so an
interrupted run resumes where it stopped.

API keys and site logins stay in settings.json on THIS computer - only
questions come down and answers go up.
"""

import json
import os

from eval_common import AnswersStore, OK_STATUSES, SettingsStore, model_slug
from hub_client import HubClient, HubError
from llm_api import ApiCallError, ModelAbort, call_api_model
from llm_judge import (
    JudgeParseError, build_judge_prompt, parse_judge_response,
    test_model_verdict,
)
from run_llms import (
    build_worklist, model_catalog, run_everything,
)

JOBS_DIR = "hub_jobs"


class JobMaster:
    """Just enough of MasterStore for the run machinery: the job's cases."""

    def __init__(self, job):
        self.cases = {}
        for case in job["cases"]:
            self.cases[case["case_id"]] = {
                "case_id": case["case_id"],
                "case_text": case["case_text"],
                "rubric": [],  # the runner never needs the rubric
                "rubric_version": case.get("rubric_version", 1),
            }


class UploadingAnswersStore(AnswersStore):
    """AnswersStore that pushes every saved answer straight to the hub.

    The local copy under hub_jobs/job_<id>/ stays as the resume point and
    audit trail; a failed upload is retried on the next save/run."""

    def __init__(self, path, images_dir, client=None, job_id=None, log=print):
        super().__init__(path, images_dir)
        self.client = client
        self.job_id = job_id
        self.log = log
        self.upload_failures = 0

    def upsert(self, record):
        super().upsert(record)
        try:
            response = self.client.upload_answer(
                self.job_id, record,
                image_paths=record.get("images", []),
                html_path=record.get("answer_html"),
            )
            if response.get("self_id_warning"):
                self.log(
                    "  Note: the answer for {} names an AI ({}) - the hub "
                    "flagged it for the blinding check.".format(
                        record["case_id"], response["self_id_warning"]
                    )
                )
        except HubError as error:
            self.upload_failures += 1
            self.log("  Upload failed for {} x {}: {} (kept locally; run the "
                     "job again to retry).".format(
                         record["case_id"], record["model_id"], error))


def hub_client_from(settings):
    hub = settings.data.get("hub", {})
    return HubClient(hub.get("url", ""), hub.get("token", ""))


def job_models(job, settings, log):
    """The catalog entries this Runner will use for the job's requested
    (site, model) pairs, with the same readiness rules as a local run.

    A model named by the job OVERRIDES the local Settings' model name,
    so two models of the same site run as two separate variants in one
    job. Legacy jobs (site only) keep using whatever this Runner is
    configured for."""
    requested = job.get("llms") or [
        {"llm_id": llm_id, "model_name": None}
        for llm_id in job.get("llm_ids", [])
    ]
    # A job asking for the test model works even when the local menu
    # option that hides it is off.
    saved_flag = settings.data["options"].get("enable_test_model", False)
    if any(entry["llm_id"] == "testmodel" for entry in requested):
        settings.data["options"]["enable_test_model"] = True
    try:
        catalog = {
            entry["model_id"]: entry for entry in model_catalog(settings)
        }
    finally:
        settings.data["options"]["enable_test_model"] = saved_flag
    chosen, skipped, seen = [], [], set()
    for wanted in requested:
        base = catalog.get(wanted["llm_id"])
        if base is None:
            skipped.append(wanted["llm_id"])
            continue
        entry = dict(base)
        model_name = (wanted.get("model_name") or "").strip()
        if model_name:
            if (entry["kind"] == "browser" and entry["model_name"]
                    and entry["model_name"] != model_name):
                log("  NOTE: this run asks {} for model '{}', but this "
                    "Runner's Settings say '{}'. Set the SITE'S OWN model "
                    "picker to '{}' before running - the answers are "
                    "recorded under that name.".format(
                        entry["display_name"], model_name,
                        entry["model_name"], model_name))
            entry["model_name"] = model_name
            entry["variant_id"] = model_slug(entry["model_id"], model_name)
            entry["scored_as"] = "{} ({})".format(
                entry["display_name"], model_name
            )
        elif entry["kind"] == "browser" and not entry["model_name"]:
            skipped.append(
                "{} (set its model name in the classic Runner's Settings "
                "first)".format(entry["display_name"])
            )
            continue
        if entry["variant_id"] in seen:
            continue
        seen.add(entry["variant_id"])
        chosen.append(entry)
    for line in skipped:
        log("  Skipping {} - this Runner is not set up for it.".format(line))
    return chosen


def run_job(job, settings, ui):
    """Run one hub job; returns (ok_count, failed_count, uploads_failed)."""
    client = hub_client_from(settings)
    master = JobMaster(job)
    workdir = os.path.join(JOBS_DIR, "job_{}".format(job["id"]))
    os.makedirs(workdir, exist_ok=True)
    answers = UploadingAnswersStore.load_or_create(
        os.path.join(workdir, "answers.json"),
        os.path.join(workdir, "answer_images"),
    )
    answers.client, answers.job_id, answers.log = client, job["id"], ui.log
    answers.upload_failures = 0

    models = job_models(job, settings, ui.log)
    if not models:
        ui.log("  Nothing this computer can do for run #{}.".format(job["id"]))
        return 0, 0, 0
    case_ids = sorted(master.cases)
    todo, skipped, failed_pairs, changed_pairs = build_worklist(
        master, answers, case_ids, models
    )
    for case_id, model in failed_pairs + changed_pairs:
        todo[model["variant_id"]].append(case_id)
    if skipped:
        ui.log("  {} answers already collected earlier are re-uploaded, not "
               "re-asked.".format(skipped))
        for (case_id, model_id), record in answers.answers.items():
            if record.get("status") in OK_STATUSES:
                answers.upsert(record)  # re-push to the hub (idempotent)
    run_everything(master, answers, settings, models, todo, ui)

    ok = failed = 0
    for model in models:
        for case_id in case_ids:
            record = answers.get(case_id, model["variant_id"])
            if record is None:
                continue
            if record.get("status") in OK_STATUSES:
                ok += 1
            else:
                failed += 1
    return ok, failed, answers.upload_failures


def finish_job(job, settings, ok, failed, uploads_failed, ui):
    client = hub_client_from(settings)
    if uploads_failed:
        ui.log("Run #{} stays open: {} upload(s) failed - start it again to "
               "retry.".format(job["id"], uploads_failed))
        return
    if ok and not failed:
        client.set_job_status(job["id"], "done", "{} answers".format(ok))
        ui.log("Run #{} is done ({} answers uploaded).".format(job["id"], ok))
    elif ok:
        client.set_job_status(
            job["id"], "done", "{} answers, {} failed".format(ok, failed)
        )
        ui.log("Run #{} finished with {} failure(s) - the failed pairs are "
               "recorded on the website.".format(job["id"], failed))
    else:
        client.set_job_status(job["id"], "failed", "no answers collected")
        ui.log("Run #{} produced no answers - marked failed on the website.".format(
            job["id"]
        ))


# ---------- LLM-as-a-judge runs ----------


def judge_entry(run, settings, log):
    """The catalog entry acting as judge (API sites and the test model
    only - a judge needs no browser), with the run's model name."""
    saved_flag = settings.data["options"].get("enable_test_model", False)
    if run["judge_llm_id"] == "testmodel":
        settings.data["options"]["enable_test_model"] = True
    try:
        catalog = {e["model_id"]: e for e in model_catalog(settings)}
    finally:
        settings.data["options"]["enable_test_model"] = saved_flag
    base = catalog.get(run["judge_llm_id"])
    if base is None or base["kind"] == "browser":
        log("  This computer cannot act as judge {} - only API models can "
            "judge.".format(run["judge_llm_id"]))
        return None
    entry = dict(base)
    entry["model_name"] = run["judge_model_name"] or entry["model_name"]
    entry["variant_id"] = model_slug(entry["model_id"], entry["model_name"])
    return entry


def run_judge_run(run, settings, ui):
    """Judge every answer in the run; returns (ok, failed, uploads_failed).
    Answers already judged (on the website or in the local resume file)
    are skipped, so an interrupted run picks up where it stopped."""
    client = hub_client_from(settings)
    entry = judge_entry(run, settings, ui.log)
    if entry is None:
        return 0, 0, 0
    workdir = os.path.join(JOBS_DIR, "judge_{}".format(run["id"]))
    os.makedirs(workdir, exist_ok=True)
    resume_path = os.path.join(workdir, "judged.json")
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            judged = set(json.load(f))
    except (OSError, ValueError):
        judged = set()
    judged |= set(run.get("done_answer_ids", []))
    answers = [a for a in run["answers"] if a["answer_id"] not in judged]
    skipped = len(run["answers"]) - len(answers)
    if skipped:
        ui.log("  {} answer(s) already judged - skipped.".format(skipped))
    ui.log("  Judge: {} - {} answer(s) to judge.".format(
        entry["variant_id"], len(answers)))
    options = settings.data["options"]
    ok = failed = uploads_failed = 0
    for position, answer in enumerate(answers, start=1):
        if ui.stop_requested:
            ui.log("  Stopped.")
            break
        rubric = answer["rubric"]
        prompt = build_judge_prompt(
            answer["case_text"], rubric, answer["response_text"],
            answer.get("image_count", 0),
        )
        payload = {"answer_id": answer["answer_id"],
                   "rubric_version": answer["rubric_version"]}
        text = ""
        try:
            if entry["kind"] == "test":
                text = test_model_verdict(len(rubric))
                thinking = "test model"
            else:
                settings_entry = dict(settings.api_model(entry["model_id"]))
                settings_entry["model"] = entry["model_name"]
                result = call_api_model(
                    entry["model_id"], settings_entry, prompt, options,
                    log=ui.log,
                )
                text = result["response_text"]
                thinking = result.get("thinking_setting", "")
            parsed = parse_judge_response(text, len(rubric))
            payload.update(
                status="ok", results=parsed["results"],
                evidence=parsed["evidence"],
                unnecessary_risk=parsed["unnecessary_risk"],
                risk_reason=parsed["risk_reason"],
                poor_approach=parsed["poor_approach"],
                poor_reason=parsed["poor_reason"],
                raw_response=text, thinking_setting=thinking,
            )
        except ModelAbort as error:
            ui.log("  {} - stopping this judge run.".format(error))
            failed += len(answers) - position + 1
            break
        except (ApiCallError, JudgeParseError) as error:
            payload.update(status="error", error=str(error), raw_response=text)
        try:
            response = client.upload_judge_grade(run["id"], payload)
        except HubError as error:
            uploads_failed += 1
            ui.log("  Upload failed for answer #{}: {}".format(
                answer["answer_id"], error))
            continue
        if payload["status"] == "ok":
            ok += 1
            judged.add(answer["answer_id"])
            with open(resume_path, "w", encoding="utf-8") as f:
                json.dump(sorted(judged), f)
            ui.log("  [{}/{}] {} x answer #{} - judged, score {}".format(
                position, len(answers), answer["case_id"],
                answer["answer_id"], response.get("score")))
        else:
            failed += 1
            ui.log("  [{}/{}] {} x answer #{} - FAILED: {}".format(
                position, len(answers), answer["case_id"],
                answer["answer_id"], payload.get("error")))
    return ok, failed, uploads_failed


def finish_judge_run(run, settings, ok, failed, uploads_failed, ui):
    client = hub_client_from(settings)
    if uploads_failed:
        ui.log("Judge run #{} stays open: {} upload(s) failed - start it "
               "again to retry.".format(run["id"], uploads_failed))
        return
    if ok and not failed:
        client.set_judge_run_status(run["id"], "done", "{} judged".format(ok))
        ui.log("Judge run #{} is done ({} answers judged) - see the AI judge "
               "page on the website.".format(run["id"], ok))
    elif ok:
        client.set_judge_run_status(
            run["id"], "done", "{} judged, {} failed".format(ok, failed))
        ui.log("Judge run #{} finished with {} failure(s).".format(
            run["id"], failed))
    else:
        client.set_judge_run_status(run["id"], "failed", "no answers judged")
        ui.log("Judge run #{} judged nothing - marked failed on the "
               "website.".format(run["id"]))


# ---------- window interface ----------


def main():
    import tkinter as tk
    from tkinter import messagebox
    import gui_common

    root = gui_common.make_root("Run AI Answers (runs from the study website)", 900, 620)
    settings = SettingsStore.load_or_create()
    hub = settings.data.setdefault("hub", {"url": "", "token": ""})

    top = tk.Frame(root)
    top.pack(fill="x", padx=8, pady=(8, 0))
    tk.Label(top, text="Hub address:").pack(side="left")
    url_var = tk.StringVar(value=hub.get("url", ""))
    tk.Entry(top, textvariable=url_var, width=34).pack(side="left", padx=4)
    tk.Label(top, text="Runner token:").pack(side="left")
    token_var = tk.StringVar(value=hub.get("token", ""))
    tk.Entry(top, textvariable=token_var, width=28, show="*").pack(
        side="left", padx=4
    )

    def save_hub():
        hub["url"] = url_var.get().strip()
        hub["token"] = token_var.get().strip()
        settings.save()
        log.log("Hub settings saved.")

    tk.Button(top, text="Save", command=save_hub).pack(side="left", padx=4)

    middle = tk.Frame(root)
    middle.pack(fill="both", expand=True, padx=8, pady=8)
    tk.Label(middle, text="Runs waiting for this computer:", anchor="w").pack(
        fill="x"
    )
    jobs_list = tk.Listbox(middle, height=6)
    jobs_list.pack(fill="x", pady=4)
    tk.Label(middle, text="AI judge runs waiting for this computer:",
             anchor="w").pack(fill="x")
    judge_list = tk.Listbox(middle, height=4)
    judge_list.pack(fill="x", pady=4)
    buttons = tk.Frame(middle)
    buttons.pack(fill="x")
    log = gui_common.LogBox(middle, height=14).pack(
        fill="both", expand=True, pady=(6, 0)
    )
    task = gui_common.BackgroundTask(root, log.log)
    current_jobs = []
    current_judge_runs = []

    def refresh_jobs():
        save_hub()
        try:
            jobs = hub_client_from(settings).jobs()
        except HubError as error:
            log.log(str(error))
            return
        current_jobs.clear()
        current_jobs.extend(jobs)
        jobs_list.delete(0, "end")
        for job in jobs:
            jobs_list.insert("end", "#{}  {} case(s)  LLMs: {}  (from {})".format(
                job["id"], len(job["cases"]), ", ".join(job["llm_ids"]),
                job["requested_by"],
            ))
        log.log("{} run(s) waiting.".format(len(jobs)))
        try:
            judge_runs = hub_client_from(settings).judge_runs()
        except HubError as error:
            judge_runs = []
            log.log("(judge runs unavailable: {})".format(error))
        current_judge_runs.clear()
        current_judge_runs.extend(judge_runs)
        judge_list.delete(0, "end")
        for run in judge_runs:
            judge_list.insert("end", "#{}  judge {}  {} answer(s)  (from {})".format(
                run["id"], run["judge_variant"], len(run["answers"]),
                run["requested_by"],
            ))
        if judge_runs:
            log.log("{} judge run(s) waiting.".format(len(judge_runs)))

    def judge_selected():
        if task.running:
            messagebox.showinfo("Busy", "A run is already going.", parent=root)
            return
        selection = judge_list.curselection()
        if not selection:
            messagebox.showinfo("Nothing selected", "Click a judge run first.",
                                parent=root)
            return
        run = current_judge_runs[selection[0]]

        def work(ui):
            ui.log("Starting judge run #{}...".format(run["id"]))
            ok, failed, uploads_failed = run_judge_run(run, settings, ui)
            finish_judge_run(run, settings, ok, failed, uploads_failed, ui)

        task.start(work, on_done=lambda r, e: (
            log.log("Problem: {}".format(e)) if e else refresh_jobs()
        ))

    def run_selected():
        if task.running:
            messagebox.showinfo("Busy", "A run is already going.", parent=root)
            return
        selection = jobs_list.curselection()
        if not selection:
            messagebox.showinfo("Nothing selected", "Click a run first.",
                                parent=root)
            return
        job = current_jobs[selection[0]]

        def work(ui):
            ui.log("Starting run #{}...".format(job["id"]))
            ok, failed, uploads_failed = run_job(job, settings, ui)
            finish_job(job, settings, ok, failed, uploads_failed, ui)

        task.start(work, on_done=lambda r, e: (
            log.log("Problem: {}".format(e)) if e else refresh_jobs()
        ))

    def stop():
        if task.running and task_ui[0] is not None:
            task_ui[0].stop_requested = True
            log.log("Stopping after the answer in flight...")

    task_ui = [None]
    original_start = task.start

    def start_capturing_ui(fn, on_done=None):
        ui = original_start(fn, on_done=on_done)
        task_ui[0] = ui
        return ui

    task.start = start_capturing_ui

    tk.Button(buttons, text="Refresh runs", command=refresh_jobs).pack(side="left")
    tk.Button(buttons, text="Start the selected run", command=run_selected).pack(
        side="left", padx=6
    )
    tk.Button(buttons, text="Start the selected judge run",
              command=judge_selected).pack(side="left", padx=6)
    tk.Button(buttons, text="Stop", command=stop).pack(side="left")

    if hub.get("url") and hub.get("token"):
        root.after(300, refresh_jobs)
    else:
        log.log("Enter the hub address (e.g. https://study.example.org) and a "
                "runner token from the hub's 'Runner tokens' page, then Save.")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
