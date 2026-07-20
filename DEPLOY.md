# Deploying the Study Hub

The hub is a small Flask + SQLite application. One person (the PI) hosts
it; graders and Runners connect over HTTPS. Everything lives in one data
folder: `study.db` plus the uploaded answer files.

## Option A - PythonAnywhere (recommended: no server administration)

1. Create an account at pythonanywhere.com (the ~$5/month plan is enough
   for a study team; it includes HTTPS).
2. Open a Bash console and clone the repository:
   `git clone https://github.com/YOUR-ACCOUNT/LLM-Medical-Cases.git`
3. `pip install --user flask`
4. Initialize the study:
   ```
   cd LLM-Medical-Cases
   export STUDYHUB_DATA=/home/YOURNAME/studyhub_data
   python -m hub.manage init-db
   python -m hub.manage create-pi "Your Name" you@example.org
   ```
5. On the Web tab: Add a new web app -> Manual configuration -> Python 3.x.
   Set the source folder to the repository, then edit the WSGI file to:
   ```python
   import os, sys
   sys.path.insert(0, "/home/YOURNAME/LLM-Medical-Cases")
   os.environ["STUDYHUB_DATA"] = "/home/YOURNAME/studyhub_data"
   from hub import create_app
   application = create_app()
   ```
6. Reload the web app. Sign in at https://YOURNAME.pythonanywhere.com,
   open **People**, and create invite links for your graders.
7. Backups: on the Tasks tab add a daily task:
   `cp -r ~/studyhub_data ~/studyhub_backup_$(date +%u)`
   (keeps a rolling week of copies).

## Option B - your own Linux server (VPS)

```
git clone ... && cd LLM-Medical-Cases
python3 -m venv venv && venv/bin/pip install flask gunicorn
export STUDYHUB_DATA=/srv/studyhub
venv/bin/python -m hub.manage init-db
venv/bin/python -m hub.manage create-pi "Your Name" you@example.org
venv/bin/gunicorn -w 2 -b 127.0.0.1:8000 'hub:create_app()'
```
Put nginx (or Caddy, which handles HTTPS automatically) in front of
port 8000. Back up `/srv/studyhub` nightly with cron.

## Connecting the Runners

Each person who runs LLMs (the PI, and any grader with their own API
keys or site logins):

1. Sign into the hub -> **Runner tokens** -> create a token (shown once).
2. On their computer, double-click **Run Hub Jobs.pyw**, paste the hub
   address and the token, Save.
3. API keys and browser-site logins are set up exactly as before, in the
   classic Runner's Settings - they never leave that computer.
4. Click **Refresh jobs**; jobs assigned to them appear; **Run the
   selected job** does the rest (browser sites still open a visible
   window and may need the person nearby, exactly as in the classic
   Runner).

## Moving an existing (desktop-programs) study into the hub

On the machine that holds `master_cases.json`, `answers.json`,
`scoring_packages/`, and any returned `scores_*.json` (with the hub's
data folder reachable, e.g. on the server after uploading the files):

```
python -m hub.manage import-legacy grader@example.org /path/to/old/study
```

Cases, answers (with their image files), and grades are imported and
owned by that grader account. The desktop programs keep working locally;
retire them whenever the team is comfortable.

## Cut-over checklist

- [ ] PI account created; graders invited and signed in
- [ ] Each runner-operator has a token and has run one test-model job
- [ ] Legacy data imported (if any); Rankings page matches the old
      Rank LLMs program's output
- [ ] Backups scheduled and one restore rehearsed
- [ ] Reminder to everyone: cases are invented vignettes - never enter
      real patient information
