# LLM Medical Cases — Evaluation Framework

A set of programs for evaluating LLM performance on medical questions. Designated
providers author clinical cases with grading rubrics, the cases are run through
multiple LLMs, providers score the LLM answers against the rubrics, and the LLMs
are ranked based on their scores.

## Project Plan

The framework consists of six programs, run in sequence:

```
> **NEW — Study Hub (web version), implemented.** The study can now run as
> a web application (`hub/`, Flask + SQLite — see `DEPLOY.md`):
> **graders** author their cases, choose which LLMs to run them through
> (running them with their own local Runner, or sending the job to the PI
> if they lack access), and grade the blinded answers in the browser;
> rubric flags and fixes propagate instantly with targeted re-grading; the
> **PI** fulfills run requests with their Runner and gets a live Elo
> ranking dashboard (with inter-rater agreement) plus CSV export and
> snapshots, and can set up an **AI judge** - an LLM grader accepted only
> where it matches the physicians' grades item by item (see below). The PI can wear both hats: their first **New case** click
> also makes them a grader, with the same blinding as everyone else.
> LLMs are still executed by the local **Hub Runner** program
> (`Run AI Answers.pyw`) because the browser-site models need a real browser
> and personal logins; API keys and site credentials never leave the
> runner's computer. `python -m hub.manage import-legacy` moves an
> existing desktop-programs study into the hub. The desktop programs below
> keep working; retire them at your own pace.

 Program 1        Program 2        Program 3        Program 4        Program 5        Program 6
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│   Case    │ em │   Case    │    │    LLM    │    │  Scoring  │ em │  Answer   │ em │    LLM    │
│  Editor   ├───►│  Merger   ├───►│  Runner   ├───►│  Package  ├───►│  Scorer   ├───►│  Ranker   │
│(provider) │ail │   (PI)    │    │   (PI)    │    │Builder(PI)│ail │ (scorer)  │ail │   (PI)    │
└───────────┘    └───────────┘    └───────────┘    └───────────┘    └───────────┘    └───────────┘
 provider_NNN_    master_          answers.json     <name>.zip       scores.json      final report
 cases.json       cases.json       + images         (blinded)
```

### Program 1 — Case Editor (`case_editor.py`) ✅ implemented

Used by each provider to create and edit cases.

- Each provider has a preassigned provider number.
- Each case gets a unique case ID of the form `PPP-CCC` (provider number +
  sequential case number), guaranteeing uniqueness even after cases from many
  providers are merged.
- A case consists of free-text case text (the clinical vignette / question) and
  one or more rubric items. **An LLM answer is considered correct only if it
  satisfies every rubric item.**
- All of a provider's cases are stored in a single JSON file
  (`provider_NNN_cases.json`) that can be emailed to the principal
  investigator (PI).
- Providers can create, edit, view, list, and delete cases.

### Program 2 — Case Merger (`merge_cases.py`) ✅ implemented

Used by the PI to combine the case files received from multiple providers into a
single master database (`master_cases.json`).

- Validates each incoming file (format version, provider number consistency,
  no duplicate case IDs, non-empty case text and rubric). All files are
  validated up front, so one bad file aborts the run before the master is
  touched.
- Merges into the master file, reporting per file what was added, updated,
  unchanged, or kept.
- Supports re-importing an updated file from a provider: for a changed case
  the PI is prompted (defaulting to the newer `updated_at` version), and
  cases the provider deleted are kept unless the PI confirms removal.
- **Mid-study rubric fixes.** Scorers can flag a rubric item as too
  difficult or wrong (the flag travels back inside their scores file);
  the PI reviews flags here and edits the rubric (reword / remove / add
  items). Every edit bumps the case's **rubric version**, is kept in a
  per-case audit history, and can be exported as a small
  `rubric_update_*.json` file to email to scorers. Their scorer program
  applies it automatically and re-queues **only the grades the change
  actually affects** (an added or reworded item re-queues the whole case;
  a removed item re-queues only answers that missed it — other grades are
  carried over unchanged). Rubric edits never re-run the LLMs (the models
  only ever see the case text), and the ranker refuses any grade made
  under an outdated rubric version, so every model on a case is always
  judged by the same rubric. Remember to tell the provider about the fix
  so their own file matches.

### Program 3 — LLM Runner (`run_llms.py`) ✅ implemented

Used by the PI to run selected cases through selected LLMs.

- **API models** (Anthropic Claude, OpenAI GPT, Google Gemini, xAI Grok) run
  unattended once an API key is entered in the in-program Settings menu.
  All selected API models run **in parallel** (one question at a time per
  provider, to stay inside each provider's rate limits), so the API phase
  takes only as long as the slowest provider.
- **Browser models with no API** (OpenEvidence, UpToDate, Doximity Ask,
  ChatGPT for Clinicians, AMBOSS, ClinicalKey AI, DynaMed, Glass Health, and
  the free no-login GPT-OSS playground at gpt-oss.com, pre-set to
  gpt-oss-120b) are
  driven through a visible browser window with Playwright; answer text is
  captured together with the **actual image files** in the answer
  (downloaded through the site's own logged-in session; an element
  screenshot is used only if a download is impossible), plus a full-page
  screenshot of every answer as an audit trail.
- **Every scored model is an (LLM, model name) pair.** API models take
  their model name from Settings (or the built-in default); for the browser
  sites you type the model name in Settings — and must, before the site can
  run (e.g. "GPT-5" for ChatGPT for Clinicians). Answers, blinded packages,
  and rankings all key on the pair, so Claude running `claude-opus-4-8` and
  Claude running a newer model are collected, scored, and ranked as two
  completely separate models. Changing the model name in Settings starts a
  fresh identity; the earlier answers stay under the old one.
- **Deep thinking, no memory.** Every model is asked to reason at length
  before answering (Claude extended/adaptive thinking — the right form is
  picked automatically from the model name, so older models like
  `claude-sonnet-4-5` and newer ones like `claude-opus-4-8` both work; GPT
  high reasoning effort; Gemini dynamic thinking; Grok 4 always reasons),
  and every case is a
  completely fresh, single-question conversation: no history is ever sent,
  no server-side storage is requested (`store: false` for OpenAI), and the
  browser sites are steered to a brand-new chat for each case. Whether deep
  thinking was on is recorded with every answer; it can be toggled in
  Settings → Options.
- Cases are selectable individually, as consecutive ranges
  (`003-001..003-020`), per provider, or all — and any selection can be
  **saved under a name and recalled later**, so the same case set can be run
  against a newly added LLM.
- Output: `answers.json` + `answer_images/` — one record per (case ID, model)
  pair with the exact prompt sent, timestamps, and model/version metadata.
  Every answer is saved as soon as it arrives, so an interrupted run resumes
  automatically (finished pairs are skipped).
- Handles rate limits and retries; a bad API key stops that model (not the
  whole run) with a plain-language message.

### Program 4 — Scoring Package Builder (`build_scoring_package.py`) ✅ implemented

Used by the PI to bundle answers into a zip file that is emailed to a scorer.
Cases and LLMs for the scorer are selected here, **independently** of what was
selected when the answers were collected.

- **Blinding:** inside the zip, answers are labeled only A, B, C..., shuffled
  per case, and image files are renamed to the blinded labels. The
  label→model key is written to a separate `*_KEY_DO_NOT_SEND.json` file
  that stays with the PI.
- Warns when an answer's text mentions an AI by name (self-identification)
  and when the case wording changed after an answer was collected.
- Offers to split packages larger than 20 MB into email-sized parts.

### Program 5 — Answer Scorer (`score_answers.py`) ✅ implemented

Used by the scorer (a provider) to grade the blinded answers in a package.

- Reads the zip from Program 4 directly; shows each answer beside its case
  text and rubric, still blinded. Answer images are pulled out of the zip
  and can be opened in the computer's normal image viewer.
- Each answer is scored **0, 1, or 2**, computed automatically from the
  scorer's y/n judgments:
  - **0** — any rubric item is missed, *or* the answer takes unnecessary
    risk with the patient (asked only when all items are covered);
  - **1** — every rubric item is covered but the approach is poor;
  - **2** — every rubric item is covered and the approach is acceptable.
  The per-item judgments, the risk and approach judgments, and an optional
  comment are all recorded alongside the score.
- Grades are saved after every answer, so the scorer can stop anytime and
  continue later; answers can also be re-graded (previous judgments become
  the defaults).
- The file is fully self-contained — the PI can email a scorer just the zip
  and this one program file, nothing else.
- Output: `scores_<package>.json` — per (case ID, label, rubric item)
  judgments plus the scorer's name, emailed back to the PI, who joins them
  to models using the package's key file (matched by `package_id`).

### Program 6 — LLM Ranker (`rank_llms.py`) ✅ implemented

Used by the PI, after scorers email back their `scores_*.json` files, to rank
the LLMs with an **Elo-type rating in which every LLM *and* every case has a
rating**. A "model" here is an (LLM, model name) pair — two model names on
the same LLM are ranked as separate entries, labeled with both (e.g.
"Anthropic Claude (claude-opus-4-8)").

- Every graded answer is one match between an LLM and a case: a score of
  **2 is a win** for the LLM, **1 is a draw**, and **0 is a loss** (the case
  beat the LLM).
- **Several scorers on the same answer**: every scorer's grade counts as
  its own match — agreement strengthens the rating, disagreement averages
  out inside the fit. Duplicate grades from the *same* scorer (a stray
  copy, or an outdated scores file next to the final one) are ignored:
  only that scorer's most recently saved grade counts. The summary line
  reports the inter-rater agreement (how often multiple scorers gave the
  same score), and each disagreement is listed.
- Ratings are not updated game-by-game like chess Elo — that would depend on
  the arbitrary order the matches are processed. Instead all ratings are
  **fitted at once by logistic regression** (maximum likelihood on the Elo
  win-probability curve), the better batch method: order-independent and
  using every match simultaneously. Each LLM and case also gets one
  imaginary drawn match against a 1500-rated opponent so perfect (or
  winless) records stay finite.
- The average case is anchored at 1500, so ratings read like chess ratings:
  the LLM-minus-case gap gives the predicted chance the LLM handles that
  case well. Higher LLM rating = stronger model; higher case rating =
  harder case.
- Joins the scores files to the PI's key files automatically (matched by
  `package_id`) and warns about grades it cannot join.
- Shows the LLM ranking (Elo, answers graded, average score, 2/1/0 counts)
  and a case-difficulty table, and exports `ranking_results/` CSVs
  (rankings, case difficulty, and every match for statistical analysis).

## The AI judge (LLM-as-a-judge) - in the Study Hub

The hub can let an LLM grade answers, but only as a *checked* stand-in
for the physicians. The design rests on five rules.

1. **The judge grades exactly like a human grader.** It receives what a
   grader sees - the case text, the rubric, and the blinded answer text
   (never the site HTML, never the model's name; self-identifying phrases
   such as "As ChatGPT..." can be redacted first) - and must return the
   same judgments: Covered / Missed for every rubric item with a verbatim
   quote as evidence and a one-sentence reason, then the unnecessary-risk
   and poor-approach questions. The 0/1/2 score is computed by the study's
   own rule (`compute_score`), never by the model. Replies must be a JSON
   object; an unreadable reply is asked for once more and then recorded
   as an error, never guessed at.
2. **Verification against the manual scores, item by item.** Every
   verdict on an answer a physician has graded (under the current rubric
   version) is compared with that grade. A verdict is **accepted only if
   every rubric item matches every physician's judgment** on that answer.
   Matching the final score is not enough: reaching 0 for the wrong item
   is rejected. The judge page reports the acceptance rate, the same-score
   rate, the answers on which the physicians themselves disagreed (the
   judge cannot match both), a per-rubric-item table of disagreements
   split into *judge lenient* (Covered where the physician said Missed)
   and *judge strict*, and every rejected verdict with the judge's
   evidence and reason beside the physician's judgment.
3. **Rubrics change; retest with one click.** Verdicts record the rubric
   version and the judge version they were made under. Editing a rubric
   item (on the case's edit page, as always) or changing the judge's
   model, instructions, or policy makes the affected verdicts *stale*:
   they leave the report, and **Test against the manual grades** re-judges
   exactly the stale and missing answers. The physicians' grades survive
   rubric edits through the usual carry-over, so the loop "sharpen the
   wording of the item the judge keeps misreading, re-test, read the
   mismatches" costs nothing on the human side. Superseded verdicts are
   kept for the record.
4. **Which LLM judges is part of the result.** A judge is an (LLM, model
   name) pair like any ranked model, plus its prompt version and extra
   instructions; all of it is recorded on every verdict. Several judges
   (different models, or the same model with different instructions) can
   be validated side by side on the same manual grades.
5. **Self-judging is a policy, not an accident.** A model grading its own
   family's answers is a known bias, so each judge has a self-judging
   policy: *skip answers from the judge's own vendor* (the default; Claude
   is Anthropic, GPT / ChatGPT for Clinicians / GPT-OSS are OpenAI, Gemini
   is Google, Grok is xAI; medical sites with undisclosed backends are
   never treated as "self"), *skip only the exact same model*, or *judge
   everything*. Under the last two, every same-vendor verdict is marked
   self-judged and the report gives the acceptance rate for self-judged
   and other answers separately, so the bias is measured rather than
   assumed. Skipped answers are listed, not silently dropped.

What the judge is used for: validation is the default and only purpose
until the PI ticks **Approved for scoring** on a judge whose acceptance
they find good enough. Even then its verdicts never enter the physicians'
ranking: **Judge every answer** also covers answers nobody has graded, and
the judge's verdicts are fitted into a *separate* ranking on the judge's
page (same Elo method, current rubric and judge version only) for
comparison with the physicians' Rankings.

Mechanics: API keys for judging are saved on the hub (they are used only
for judging; the keys that run the AIs on cases stay in each person's
Runner as before). A run judges answers one by one in the background,
saving each verdict as it arrives, with a Stop button; if the site is
restarted mid-run, `python -m hub.manage judge-run RUN_ID` resumes it
(finished verdicts are kept). A built-in keyword *test judge* needs no key
and exercises the whole pipeline - it is deliberately naive, and watching
validation reject it is a good first demonstration. Verdict pages
identify answers by number, not by model, so a PI who also grades is not
unblinded by reading the judge's reasoning.

## Data Formats

All programs exchange plain JSON files so they are easy to email, inspect, and
version. The provider case file produced by Program 1 looks like:

```json
{
  "format_version": 1,
  "provider_number": 3,
  "max_assigned_case_number": 1,
  "cases": [
    {
      "case_id": "003-001",
      "case_number": 1,
      "case_text": "A 54-year-old man presents with crushing chest pain...",
      "rubric": [
        "Identifies acute myocardial infarction as the leading diagnosis",
        "Recommends immediate ECG",
        "Recommends aspirin administration"
      ],
      "created_at": "2026-07-11T12:00:00Z",
      "updated_at": "2026-07-11T12:00:00Z"
    }
  ]
}
```

## Getting started on Microsoft Windows

Every program is a normal **window-based Windows application** — no command
line, no options to remember.

1. Install Python once from [python.org](https://www.python.org/downloads/)
   (click "Install Now"; the defaults include everything these programs
   need, including the window system).
2. Put the program files in a folder, and **double-click the launcher** for
   the program you need:

   | Double-click | Who uses it |
   |---|---|
   | `Case Editor.pyw` | each provider |
   | `Merge Cases.pyw` | the PI |
   | `Run LLMs.pyw` | the PI |
   | `Build Scoring Package.pyw` | the PI |
   | `Score Answers.pyw` | each scorer |
   | `Rank LLMs.pyw` | the PI |

   The `.pyw` launchers open the window without a console behind it. All
   data files are created in the same folder, so keep each person's
   programs in one folder (providers only need the Case Editor files;
   scorers only need `score_answers.py` next to their zip).

## Program 1 (for providers) — double-click `Case Editor.pyw`

On first run a small dialog asks for your preassigned provider number and the
file `provider_NNN_cases.json` is created; on later runs it is found and
opened automatically. The window shows your case list on the left; on the
right you type the case text and the rubric (**one rubric item per line**),
then click **Save case**. New case / Delete case buttons are under the list,
and the program warns before anything unsaved is lost. When your cases are
ready, email your `provider_NNN_cases.json` file to the principal
investigator.

## Program 2 (for the PI) — double-click `Merge Cases.pyw`

If the `provider_NNN_cases.json` files the providers emailed you are not
already in the study folder, the program asks where they are when it starts
(a normal folder-chooser box — your Downloads folder, for example), and a
**Change folder...** button switches folders anytime. The window lists the
provider files it finds there; select some (Ctrl-click) and click **Merge
selected**, or just **Merge all**. The
master database `master_cases.json` is created in the same folder and a log
pane shows exactly what was added, updated, or kept. Merging the same file
twice is safe. When a provider re-sends an updated file, a dialog shows both
versions of each changed case and asks which to keep (noting which is
newer); cases missing from the new file are kept unless you confirm their
removal. **View the master database** shows everything in a table.

## Program 3 (for the PI) — double-click `Run LLMs.pyw`

The window has four tabs. On **Run**: step 1, pick the cases (all / typed
ranges like `003-001..003-020` / a saved set — and any selection can be saved
under a name for reuse); step 2, tick the LLMs (each line shows whether it is
ready and how many of the chosen cases it has already answered); step 3,
click **Start the run**. Progress streams into the log pane; already-answered
pairs are skipped automatically, previously failed pairs are retried if the
checkbox is on, and **Stop after the current answer** stops cleanly (nothing
is lost — every answer is saved the moment it arrives).

**Case sets** manages the saved selections. **Answers so far** is a table of
every collected answer (double-click-free: select a row and click View).
**Settings** holds the API keys (entered hidden, shown last-4 only), the site
logins, a one-click **Browser automation setup** (installs Playwright with
your consent — a few-hundred-MB one-time download), **Log in now** for each
site (a real browser window opens; you complete the login including any
verification code or CAPTCHA yourself — never automated — and it is
remembered), and an **Options** dialog (timeouts, pacing, deep thinking, the
fake test model). During the browser phase stay at the computer: if a site
misbehaves, a dialog offers retry / do-it-by-hand (the program still captures
text, images, and a screenshot) / skip / set the site aside. A replacement
`site_selectors.json` next to the program fixes a site redesign without code
changes.

**If Google sign-in says "this browser or app may not be secure":** Google
sometimes refuses its sign-in inside an automated browser. The program
minimizes this (it prefers your real installed Chrome or Edge and does not
advertise automation), but if Google still refuses, simply use the site's own
email-and-password or emailed-code sign-in instead of the "Continue with
Google" button. It is a one-time step — the login is remembered in that
site's browser profile afterwards.

Please note: automated querying of the LLM sites (OpenEvidence, UpToDate,
Doximity, ChatGPT for Clinicians, AMBOSS, ClinicalKey AI, DynaMed, Glass
Health, gpt-oss.com) happens under your own accounts and is your responsibility under
those services' terms. Keep `settings.json` private.

## Program 4 (for the PI) — double-click `Build Scoring Package.pyw`

Select the cases in the table (Select all, Ctrl-click, or type a range and
click Apply), tick the AIs to include, name the package, and click **Build
the package** — these choices are independent of how the answers were
collected, so different scorers can get different slices. Coverage gaps
raise a dialog (include as-is / exclude / cancel), and oversized packages
offer to split into email-sized parts. Email `scoring_packages/<name>.zip`
to the scorer. The matching `<name>_KEY_DO_NOT_SEND.json` reveals which AI
wrote each answer: it stays with you and is needed later by the ranker.
**Never send the key file to a scorer.**

## Program 5 (for the scorer) — double-click `score_answers.py`

Only `score_answers.py` is needed — the program is fully self-contained.
When it starts it asks **where the zip you were emailed is saved** (a normal
folder-chooser box — point it at your Downloads folder, for example), finds
the package there, and asks your name once. The window shows the answer list
on the left (with each score as you go) and, on the right, the case text,
the anonymized answer, an **Open the images** button when the answer has
figures, and the rubric with **Covered / Missed** buttons per item. When
every item is covered, the two follow-up questions (unnecessary risk? poor
approach?) light up, the 0/1/2 score is shown live with its reason, and
**Save grade + next** moves on. Every grade is saved instantly to
`scores_<package>.json` **in the same folder as the zip**; close the window
anytime and reopen later to continue, or click any answer in the list to
re-grade it (previous judgments are pre-filled). When everything is graded,
email the scores file back to the PI.

## Program 6 (for the PI) — double-click `Rank LLMs.pyw`

When it starts, the program asks **where you saved the `scores_*.json`
files the scorers emailed back** (a normal folder-chooser box; if they are
already in the study folder, cancelling just uses that). Your key files are
found automatically in the study folder's `scoring_packages/` — and in the
chosen folder too, if you keep everything in one place. The window then
opens with the ranking already fitted: an **LLM rankings** tab (Elo, answers
graded, average score, 2/1/0 counts, and the predicted chance of handling an
average case well) and a **Case difficulty** tab (higher Elo = harder).
Anything that could not be joined to a key is listed as a warning. **Export
CSV files for analysis** writes `ranking_results/llm_rankings.csv`,
`case_difficulty.csv`, and `matches.csv`. Reopen it whenever another scores
file arrives — it always refits from everything present.

## Files created alongside the programs

| File / folder | Created by | Notes |
|---|---|---|
| `provider_NNN_cases.json` | Program 1 | email to the PI |
| `master_cases.json` | Program 2 | the PI's case database |
| `settings.json` | Program 3 | API keys and logins — **keep private** |
| `case_sets.json` | Programs 3/4 | saved case selections |
| `answers.json`, `answer_images/` | Program 3 | collected answers |
| `browser_profiles/` | Program 3 | remembered site logins |
| `scoring_packages/` | Program 4 | zips to email + key files to keep |
| `scores_<package>.json` | Program 5 | the scorer's grades — email back to the PI |
| `ranking_results/` | Program 6 | ranking, case-difficulty, and match CSVs |
| `ranking_results/` | Program 6 | ranking and match CSVs |

## Tests

```
python3 -m unittest discover -p "test_*.py"
```

(The data logic is fully covered by the tests; the windows were additionally
exercised end-to-end under a virtual display during development.)
