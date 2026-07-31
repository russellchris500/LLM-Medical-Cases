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

## Option B - your own server (VPS), step by step

A "VPS" is a small rented computer in a data center that is always on.
Total cost: about $5-6/month for the server plus ~$10/year for a domain
name. Written for a first-time deployer; every command is copy-paste.

### B1. Rent the server and a domain

1. Create an account at DigitalOcean, Hetzner, or Linode (all fine; the
   smallest plan is plenty). Create a server ("droplet" on DigitalOcean)
   with **Ubuntu 24.04 LTS**. Choose password login if SSH keys are
   unfamiliar. Note the server's IP address (e.g. 203.0.113.7).
2. Buy a domain (Namecheap, Porkbun, Cloudflare - any registrar), e.g.
   `yourstudy.org`. In the registrar's DNS settings add an **A record**:
   name `hub` (or `@`), value = your server's IP. Your site will be
   `https://hub.yourstudy.org`. (The domain is what makes automatic
   HTTPS possible.)

### B2. First login and basics

On Windows, open PowerShell (ssh is built in):

```
ssh root@203.0.113.7
```

Then on the server:

```
apt update && apt -y upgrade
apt -y install python3-venv git ufw
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw --force enable
apt -y install unattended-upgrades        # OS security updates, automatic
```

### B3. Install the hub

```
cd /srv
git clone https://YOUR-GITHUB-USERNAME:YOUR-TOKEN@github.com/russellchris500/LLM-Medical-Cases.git
cd LLM-Medical-Cases
python3 -m venv venv
venv/bin/pip install flask gunicorn
export STUDYHUB_DATA=/srv/studyhub_data
venv/bin/python -m hub.manage init-db
venv/bin/python -m hub.manage create-pi "Your Name" you@example.org
```

(The GitHub token is a fine-grained personal access token with read
access to the repository - see the PythonAnywhere section.)

### B4. Run it as a service (starts on boot, restarts if it crashes)

Create `/etc/systemd/system/studyhub.service` (e.g. `nano` that path)
with:

```ini
[Unit]
Description=LLM Medical Cases Study Hub
After=network.target

[Service]
WorkingDirectory=/srv/LLM-Medical-Cases
Environment=STUDYHUB_DATA=/srv/studyhub_data
ExecStart=/srv/LLM-Medical-Cases/venv/bin/gunicorn -w 2 -b 127.0.0.1:8000 'hub:create_app()'
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:

```
systemctl daemon-reload
systemctl enable --now studyhub
systemctl status studyhub        # should say "active (running)"
```

### B5. HTTPS with Caddy (automatic certificates)

Caddy is a web server that obtains and renews the HTTPS certificate for
you - zero certificate maintenance:

```
apt -y install debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt update && apt -y install caddy
```

Replace the contents of `/etc/caddy/Caddyfile` with (your domain):

```
hub.yourstudy.org {
    reverse_proxy 127.0.0.1:8000
}
```

Then `systemctl reload caddy`. Wait a minute for the certificate, and
open https://hub.yourstudy.org - the sign-in page should appear with a
padlock. Invite graders from the People page as usual.

### B6. Backups (do not skip)

`crontab -e` and add:

```
15 3 * * * cp -r /srv/studyhub_data /srv/backup_$(date +\%u)
```

That keeps seven rotating daily copies ON the server. Also keep an
OFF-server copy: from your own PC, occasionally run

```
scp -r root@203.0.113.7:/srv/studyhub_data ./studyhub_backup
```

(or enable the provider's automated snapshots - usually ~$1/month).

### B7. Updating the hub later

```
cd /srv/LLM-Medical-Cases && git pull && systemctl restart studyhub
```

### PythonAnywhere or VPS?

- **PythonAnywhere**: no server to maintain, HTTPS included, but the
  address is YOURNAME.pythonanywhere.com unless you pay for a custom
  domain, and you are limited to what the platform allows.
- **VPS**: your own domain, full control, marginally cheaper at scale -
  but OS updates, firewall, and backups are your responsibility (the
  steps above automate almost all of it).

## The PI can be a grader too

One account, both hats: the first time the PI clicks **New case** they
are also given a grader number (or run
`python -m hub.manage make-grader pi@example.org`). From then on the PI
authors, runs, and grades their own cases exactly like any grader -
blinded the same way - while keeping all the PI pages (Rankings,
Cross-grading, People, the run queue).

## Connecting the Runners

Each person who runs LLMs (the PI, and any grader with their own API
keys or site logins):

1. Sign into the hub -> **Runner tokens** -> create a token (shown once).
2. On their computer, double-click **Run AI Answers.pyw**, paste the hub
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
