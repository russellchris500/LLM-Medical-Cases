"""The Hub Runner: runs Study Hub jobs on this computer.

Double-click "Run Hub Jobs.pyw". The program signs into the hub with the
runner token from Settings, lists the run jobs assigned to you (your own
jobs, or - for the PI - jobs graders sent over), and runs them with the
exact same machinery as the classic Runner: API models in parallel,
browser sites in a visible window with you nearby. Every answer is
uploaded the moment it is captured, so an interrupted job resumes where
it stopped.

API keys and site logins stay in settings.json on THIS computer - only
questions come down and answers go up.
"""

import json
import os

from eval_common import AnswersStore, OK_STATUSES, SettingsStore
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
    """The catalog entries this Runner will use for the job's llm_ids,
    with the same readiness rules as a local run."""
    wanted = list(job["llm_ids"])
    # The job asked for the test model explicitly: make sure the catalog
    # includes it even when the local menu option that hides it is off.
    saved_flag = settings.data["options"].get("enable_test_model", False)
    if "testmodel" in wanted:
        settings.data["options"]["enable_test_model"] = True
    try:
        catalog = model_catalog(settings)
    finally:
        settings.data["options"]["enable_test_model"] = saved_flag
    chosen, skipped = [], []
    for entry in catalog:
        if entry["model_id"] not in wanted:
            continue
        if entry["kind"] == "browser" and not entry["model_name"]:
            skipped.append(
                "{} (set its model name in the classic Runner's Settings "
                "first)".format(entry["display_name"])
            )
            continue
        chosen.append(entry)
    known = {entry["model_id"] for entry in chosen}
    for llm_id in wanted:
        if llm_id not in known and not any(llm_id in s for s in skipped):
            skipped.append(llm_id)
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
        ui.log("  Nothing this Runner can do for job #{}.".format(job["id"]))
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
        ui.log("Job #{} stays open: {} upload(s) failed - run it again to "
               "retry.".format(job["id"], uploads_failed))
        return
    if ok and not failed:
        client.set_job_status(job["id"], "done", "{} answers".format(ok))
        ui.log("Job #{} is done ({} answers uploaded).".format(job["id"], ok))
    elif ok:
        client.set_job_status(
            job["id"], "done", "{} answers, {} failed".format(ok, failed)
        )
        ui.log("Job #{} finished with {} failure(s) - the failed pairs are "
               "recorded on the hub.".format(job["id"], failed))
    else:
        client.set_job_status(job["id"], "failed", "no answers collected")
        ui.log("Job #{} produced no answers - marked failed on the hub.".format(
            job["id"]
        ))


# ---------- window interface ----------


def main():
    import tkinter as tk
    from tkinter import messagebox
    import gui_common

    root = gui_common.make_root("Hub Runner (runs Study Hub jobs)", 900, 620)
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
    tk.Label(middle, text="Run jobs waiting for this Runner:", anchor="w").pack(
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
        log.log("{} open job(s).".format(len(jobs)))

    def run_selected():
        if task.running:
            messagebox.showinfo("Busy", "A job is already running.", parent=root)
            return
        selection = jobs_list.curselection()
        if not selection:
            messagebox.showinfo("Nothing selected", "Click a job first.",
                                parent=root)
            return
        job = current_jobs[selection[0]]

        def work(ui):
            ui.log("Running job #{}...".format(job["id"]))
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

    tk.Button(buttons, text="Refresh jobs", command=refresh_jobs).pack(side="left")
    tk.Button(buttons, text="Run the selected job", command=run_selected).pack(
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
