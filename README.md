# LLM Medical Cases — Evaluation Framework

A set of programs for evaluating LLM performance on medical questions. Designated
providers author clinical cases with grading rubrics, the cases are run through
multiple LLMs, providers score the LLM answers against the rubrics, and the LLMs
are ranked based on their scores.

## Project Plan

The framework consists of six programs, run in sequence:

```
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

### Program 3 — LLM Runner (`run_llms.py`) ✅ implemented

Used by the PI to run selected cases through selected LLMs.

- **API models** (Anthropic Claude, OpenAI GPT, Google Gemini, xAI Grok) run
  unattended once an API key is entered in the in-program Settings menu.
- **Browser models with no API** (OpenEvidence, UpToDate, Doximity GPT) are
  driven through a visible browser window with Playwright; answer text *and
  images* are captured, along with a full-page screenshot of every answer.
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

### Program 5 — Answer Scorer (`score_answers.py`) — planned

Used by the scorer (a provider) to grade the blinded answers in a package.

- Reads the zip from Program 4 directly; shows each answer beside its case
  text and rubric, still blinded.
- For each rubric item the scorer marks met / not met; a case answer is
  correct only if every rubric item is met.
- Output: `scores.json` — per (case ID, label, rubric item) judgments, emailed
  back to the PI, who joins them to models using the package's key file.

### Program 6 — LLM Ranker (`rank_llms.py`) — planned

Used by the PI to aggregate scores and rank the LLMs.

- Joins `scores.json` files to the key files, then computes per-LLM metrics:
  cases fully correct (all rubric items met), fraction of rubric items met,
  breakdowns by provider and by case.
- Produces a summary table and CSV export for statistical analysis.

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

All programs are fully menu-driven — no command-line options to remember.
You just run the program and answer its questions.

## Program 1 Usage (for providers)

Requires Python 3.8+ (standard library only — nothing to install).

```
python3 case_editor.py
```

On first run you are asked for your preassigned provider number, and the file
`provider_NNN_cases.json` is created. On later runs the program finds that
file in the same folder and loads it automatically (if several case files are
present it shows a numbered list and asks which one to open). A menu offers:

- **N** — create a new case (enter case text, then rubric items one per line)
- **L** — list all cases
- **V** — view a case in full
- **E** — edit a case (replace case text; add, edit, or remove rubric items)
- **D** — delete a case
- **Q** — quit (the file is saved after every change, so quitting is always safe)

When your cases are ready, email your `provider_NNN_cases.json` file to the
principal investigator.

## Program 2 Usage (for the PI)

Requires Python 3.8+ (standard library only). Copy the `provider_NNN_cases.json`
files the providers emailed you into one folder and run the program there:

```
python3 merge_cases.py
```

The master database `master_cases.json` is created in the same folder on the
first merge and reloaded on later runs. The menu offers:

- **M** — merge provider files: the program lists the provider files it finds
  in the folder and you pick which to merge (e.g. `1,3`, or `A` for all)
- **L** — list the master database, grouped by provider
- **Q** — quit (the master is saved after every merge)

Merging the same file twice is safe (already-merged cases are reported as
unchanged). When a provider re-sends an updated file, the program shows both
versions of each changed case and asks which to keep, suggesting the newer
one; cases missing from the new file (deleted by the provider) are kept
unless you confirm their removal.

## Program 3 Usage (for the PI)

Run it in the same folder as `master_cases.json`:

```
python3 run_llms.py
```

The menu offers **R**un, **C**ase sets, **A**nswers so far, **S**ettings, and
**Q**uit. A run has three steps: choose the cases (all / typed IDs and ranges
like `003-001..003-020` / one provider / a saved set — typed selections can be
saved under a name for reuse), choose the LLMs (the list shows which are ready
and how many of the chosen cases each has already answered), then confirm.
Already-answered pairs are skipped automatically; previously failed pairs are
offered for retry.

**Settings** holds the API keys (typed with hidden input, shown last-4 only)
and the site logins. `settings.json` is saved with private permissions —
keep it out of email and version control.

**One-time setup for the browser models:** Settings → "Browser automation
setup" installs Playwright and a browser for it to drive (a few hundred MB,
with your consent). Then use "Log in now" under each site: a browser window
opens, you finish the login yourself — including any verification code or
CAPTCHA, which the program never automates — and the login is remembered for
future runs. During a browser run you should stay at the computer; if a site
misbehaves, the program offers to retry, let you drive that case by hand
(it still captures the text, images, and a screenshot), skip it, or set the
site aside. If a site changes its design, a replacement `site_selectors.json`
file placed next to the program fixes the automation without code changes.

Please note: automated querying of subscription sites (OpenEvidence, UpToDate,
Doximity) happens under your own accounts and is your responsibility under
those services' terms.

## Program 4 Usage (for the PI)

```
python3 build_scoring_package.py
```

Choose the cases (from those that have answers) and the AIs to include —
these choices are independent of how the answers were collected, so you can
send different scorers different slices. The program checks coverage (cases
missing an answer from some AI can be included as-is or excluded), then
builds `scoring_packages/<name>.zip` — email that file to the scorer. The
matching `<name>_KEY_DO_NOT_SEND.json` reveals which AI wrote each answer:
it stays with you and is needed later by the ranker. **Never send the key
file to a scorer.**

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

## Tests

```
python3 -m unittest discover -p "test_*.py"
```
