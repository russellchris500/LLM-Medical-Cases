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
    jobs_list = tk.Listbox(middle, height=8)
    jobs_list.pack(fill="x", pady=4)
    buttons = tk.Frame(middle)
    buttons.pack(fill="x")
    log = gui_common.LogBox(middle, height=14).pack(
        fill="both", expand=True, pady=(6, 0)
    )
    task = gui_common.BackgroundTask(root, log.log)
    current_jobs = []

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
