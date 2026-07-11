# LLM Medical Cases — Evaluation Framework

A set of programs for evaluating LLM performance on medical questions. Designated
providers author clinical cases with grading rubrics, the cases are run through
multiple LLMs, providers score the LLM answers against the rubrics, and the LLMs
are ranked based on their scores.

## Project Plan

The framework consists of five programs, run in sequence:

```
 Program 1            Program 2            Program 3            Program 4            Program 5
┌────────────┐       ┌────────────┐       ┌────────────┐       ┌────────────┐       ┌────────────┐
│   Case     │ email │   Case     │       │    LLM     │       │   Answer   │       │    LLM     │
│  Editor    ├──────►│  Merger    ├──────►│   Runner   ├──────►│   Scorer   ├──────►│  Ranker    │
│ (provider) │       │    (PI)    │       │    (PI)    │       │ (provider) │       │    (PI)    │
└────────────┘       └────────────┘       └────────────┘       └────────────┘       └────────────┘
 provider_NNN_        master_cases.json    answers.json         scores.json          final report
 cases.json
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

### Program 3 — LLM Runner (`run_llms.py`) — planned

Used by the PI to run selected cases through multiple LLMs.

- Reads `master_cases.json` and a config file listing the LLMs to query
  (model name, API provider, key, temperature, etc.).
- Sends each case's case text to each configured LLM with a standardized
  prompt, and records the raw answers.
- Output: `answers.json` — one record per (case ID, model) pair, including
  timestamps and model/version metadata for reproducibility.
- Handles rate limits and retries; can resume an interrupted run.

### Program 4 — Answer Scorer (`score_answers.py`) — planned

Used by providers to grade the LLM answers against the rubrics.

- Presents each answer alongside its case text and rubric, **blinded to which
  LLM produced it** (answers are shown in random order under anonymous labels)
  to avoid bias.
- For each rubric item the scorer marks met / not met; a case answer is
  correct only if every rubric item is met.
- Output: `scores.json` — per (case ID, model, rubric item) judgments plus the
  overall correct/incorrect result per answer.

### Program 5 — LLM Ranker (`rank_llms.py`) — planned

Used by the PI to aggregate scores and rank the LLMs.

- Per-LLM metrics: cases fully correct (all rubric items met), fraction of
  rubric items met, breakdowns by provider and by case.
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

## Program 1 Usage

Requires Python 3.8+ (standard library only — nothing to install).

```
python3 case_editor.py            # uses/creates the case file in the current directory
python3 case_editor.py mycases.json   # or specify a file explicitly
```

On first run you are asked for your preassigned provider number, and the file
`provider_NNN_cases.json` is created. On later runs the same file is loaded
automatically. A menu offers:

- **N** — create a new case (enter case text, then rubric items one per line)
- **L** — list all cases
- **V** — view a case in full
- **E** — edit a case (replace case text; add, edit, or remove rubric items)
- **D** — delete a case
- **Q** — quit (the file is saved after every change, so quitting is always safe)

When your cases are ready, email your `provider_NNN_cases.json` file to the
principal investigator.

## Program 2 Usage

Requires Python 3.8+ (standard library only). Run it in the folder where you
keep `master_cases.json` (it is created on the first merge).

```
python3 merge_cases.py provider_003_cases.json provider_007_cases.json
python3 merge_cases.py --yes provider_003_cases.json    # no prompts: newer version wins
python3 merge_cases.py --yes --prune provider_003_cases.json  # also drop provider-deleted cases
python3 merge_cases.py --list                            # show the master's contents
python3 merge_cases.py -m study2_master.json --list      # use a different master file
```

Merging the same file twice is safe (already-merged cases are reported as
unchanged). When a provider re-sends an updated file, changed cases trigger a
confirmation prompt that defaults to the newer version; cases missing from the
new file (deleted by the provider) are kept unless removal is confirmed or
`--prune` is given.

## Tests

```
python3 -m unittest test_case_editor.py test_merge_cases.py
```
