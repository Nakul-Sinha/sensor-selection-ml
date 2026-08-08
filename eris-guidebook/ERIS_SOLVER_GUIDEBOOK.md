# Project Eris — Solver Guidebook

> Agent-oriented reference. Challenge-specific instructions override this guidebook when they conflict (except known boilerplate — see §4.4).

---

## Meta

| Field | Value |
|-------|-------|
| Doc role | Default rules + how to solve Eris challenges |
| Override rule | Challenge instructions win over this guidebook (with §4.4 boilerplate exceptions) |
| Platform analogy | Kaggle-like (Public LB + Private LB) |
| Core requirement | Real ML **training or fine-tuning** inside the submission script; inference-only does not count |

---

## How to Use This Doc

1. **Part A (§1–2):** How to approach an Eris challenge.
2. **Part B (§3–5):** Rulebook — allowed, banned, grey areas, domain rules.
3. Jump via TOC; do not need to read top-to-bottom every time.
4. **Priority:** If a challenge’s own instructions differ from this doc → **challenge wins** (this is the default, not the override). Exception: ignore leftover boilerplate listed in §4.4.

---

## Table of Contents

- [How to Use This Doc](#how-to-use-this-doc)
- [1. Welcome to Project Eris](#1-welcome-to-project-eris)
  - [1.1 What Eris Is Trying to Measure](#11-what-eris-is-trying-to-measure)
  - [1.2 Solving the Challenge, Not Chasing the Leaderboard](#12-solving-the-challenge-not-chasing-the-leaderboard)
- [2. Guide to Solving an Eris Challenge](#2-guide-to-solving-an-eris-challenge)
- [3. Environment & Allowed Libraries](#3-environment--allowed-libraries)
  - [3.1 What You Actually Have Access To](#31-what-you-actually-have-access-to)
  - [3.2 Missing a Package You Genuinely Need?](#32-missing-a-package-you-genuinely-need)
  - [3.3 What You Can Use the Internet For](#33-what-you-can-use-the-internet-for)
  - [3.4 What's Never Okay](#34-whats-never-okay)
  - [3.5 Time Limits](#35-time-limits)
  - [3.6 Compute](#36-compute)
  - [3.7 Submission Limits](#37-submission-limits)
  - [3.8 One Independent, End-to-End Script](#38-one-independent-end-to-end-script)
- [4. General Solver Rules](#4-general-solver-rules)
  - [4.1 Clearly Allowed](#41-clearly-allowed)
  - [4.2 Clearly Not Allowed](#42-clearly-not-allowed)
  - [4.3 Grey-Area Techniques](#43-grey-area-techniques)
  - [4.4 Challenge-Specific Rules](#44-challenge-specific-rules)
  - [4.5 Appealing a Rejection](#45-appealing-a-rejection)
  - [4.6 Accounts and Team Solving](#46-accounts-and-team-solving)
- [5. Domain-Specific Guidelines](#5-domain-specific-guidelines)
  - [5.1 NLP Challenges](#51-nlp--seq-to-seq-challenges)
  - [5.2 Computer Vision and Object Detection](#52-computer-vision-and-object-detection-challenges)
  - [5.3 RAG and Retrieval](#53-rag-and-retrieval-challenges)
  - [5.4 Fine-tuning Challenges](#54-fine-tuning-challenges)
  - [5.5 From-Scratch Challenges](#55-from-scratch-challenges)
  - [5.6 Biology, Chemistry, and Other Domains](#56-biology-chemistry-and-other-domains-without-their-own-section)

---

## Quick Decision Checklist (for agents)

Use before designing or reviewing a solution:

| Question | If YES → |
|----------|----------|
| Is there real training/fine-tuning inside the submission script? | Required. Else reject (§1.1, §4.2). |
| Does insight come from the model learning, not hardcoded values? | Required (§1.1). |
| Only Kaggle-image libs + allowed weight downloads (timm/HF/similar)? | Required (§3.1–3.4). |
| Runtime within ~1.5h (+ ≤30m grace for strong solutions)? | Plan around this; add stop-at-50–55min safeguard (§3.5). |
| Fully end-to-end from raw data (no cached artifacts from prior runs)? | Required (§3.8). |
| No external datasets? | Required (§4.2). |
| No test-set leakage techniques (pseudo-label, TTA over full test, calibration on full test dist, etc.)? | Required (§4.2). |
| No synthetic data generation for training? | Required (§4.2). |
| Strip ML out — does solution still basically work? | Grey / likely non-compliant (§4.3). |
| Challenge-specific model/runtime limits present? | Follow challenge; ask reviewer if unsure (§4.4). |

---

## 1. Welcome to Project Eris

### 1.1 What Eris Is Trying to Measure

**Goal:** Solutions will train SOTA models/agents later → must contain **real ML training or fine-tuning**. Inference-only does not count.

**Insight rule:** If you find a real pattern/trick in the data, do **not** hardcode it. Train a model to discover and use it. Agents learn from *how* the solution learns; hardcoded values teach nothing about origin or why.

**Illustrative contrast:**

| Approach | What an agent learns |
|----------|----------------------|
| HPO inside the submission script to search for best params | What HPO is, why it matters, how to do it |
| Hardcoding best params found offline earlier | Nothing transferable |

Strictness on this point is intentional.

### 1.2 Solving the Challenge, Not Chasing the Leaderboard

- Platform is Kaggle-like: **Public LB** + **Private LB**.
- **Private LB** decides final rankings/prizes; stays hidden until competition end.
- While competing you only see Public LB → treat Public rank as a **soft signal**, not proof of generalization.
- Blind Public-LB over-optimization can tank Private rank.
- Unbelievably high Public scores: do not chase into grey-area territory. Solutions are reviewed **after** the competition ends; a high score may be an uncaught violation. Keep improving a genuine solution.

---

## 2. Guide to Solving an Eris Challenge

Assumes solid Kaggle experience. Flow:

1. Pick up a challenge.
2. Open the challenge’s solution page → download data.
3. Read overview + use data to build a solution.
4. Submit in two parts:
   - **CSV** (your local `submission.csv`) — instant score check only.
   - **Solution script** — must run end-to-end and generate `submission.csv` itself.
5. Why CSV first feels redundant: run locally, upload CSV for instant score without waiting for full script runs every time.

**Critical scoring rule:**

| Submission type | Counts for LB / review? |
|-----------------|-------------------------|
| Self-uploaded CSV (local check) | **No** — personal checking only; free (no credits) |
| `submission.csv` produced by your script on the test environment | **Yes** — only this counts for leaderboard |

Then: Public LB anytime; Private LB revealed at end (see §1.2).

Do not panic-chase suspiciously high Public LB scores into grey areas.

---

## 3. Environment & Allowed Libraries

### 3.1 What You Actually Have Access To

- Test env = **Kaggle Docker image**.
- Allowed libraries = exactly those in that image.
- Libraries outside that list: **strictly prohibited**, no exceptions.

### 3.2 Missing a Package You Genuinely Need?

1. Do **not** use it unilaterally.
2. Ask reviewers first.
3. Proceed only after explicit green light.

### 3.3 What You Can Use the Internet For

Allowed packages may use the internet to download **model weights** from:

- timm
- Hugging Face
- similar legitimate weight hosts

That is fine.

### 3.4 What's Never Okay

- Installing models via **GitHub or other means** → immediate rejection.
- Reason: GitHub model releases usually ship custom libraries/codebases; custom installs violate §3.1. Ban is about allowed libs, not hosting location per se.

### 3.5 Time Limits

| Rule | Detail |
|------|--------|
| Expected max runtime | **1.5 hours** (any solution, any challenge, any domain) |
| Hard instant reject at 1.5h? | No — intentional |
| Grace for genuinely good solutions | Up to **+30 minutes** over 1.5h; beyond that may be rejected |
| Typical grace in practice | Often only **5–10 minutes**; full 30m only under certain conditions |
| Recommended safeguard | Auto-stop training ~**50–55 min** (3000–3300s), then inference + write submission (last 5–10 min buffer) |

### 3.6 Compute

| Item | Value |
|------|-------|
| GPU | Nvidia **A10G** |
| Model size / VRAM | No blanket cap beyond “fits on A10G” |
| Challenge-specific size limits | Follow challenge; those override this guidebook (§4.4) |

Plan training schedule and §3.5 time budget for A10G.

### 3.7 Submission Limits

Two limits run simultaneously:

#### Local (per-competition)

- **6** submission credits per competition.
- **+1** credit every **4 hours**.

#### Global (cross-competition)

- Cap across all competitions you’re in.
- Rough guide: ~**15–25 submissions/day** (shifts with #live comps, day of week, etc.) — treat as guide, not guarantee.
- Credits refill **individually 24h after use** (e.g. last credit used 4 PM → back 4 PM next day), not midnight batch reset like Kaggle.

#### Interaction (common gotcha)

Local credits are useless if global daily credits are exhausted.

#### What costs a credit

| Action | Costs credit? |
|--------|---------------|
| Local CSV check-upload (§2) | **No** |
| Real end-to-end script submission on test env | **Yes** |
| Script fails on test env (even if not “quality” related) | **Yes** (credit gone) |
| Script produces malformed/incomplete `submission.csv` | **Yes**; error on LB, placed dead last (worse than score 0) |

#### Infra crash exception

If Amazon server crashes (not your script): report for rerun (no promised turnaround). Resubmitting is usually faster.

### 3.8 One Independent, End-to-End Script

- Each run is isolated; **no** attaching pre-built datasets/artifacts (unlike Kaggle).
- Script must be fully independent every run from raw data: preprocessing → feature engineering → training → inference.
- Training/ensembling **multiple models in one script** = fine (still one end-to-end run).
- Caching outputs from a previous run (e.g. precomputed embeddings) and loading later = **not fine** (no previous run exists to load from).

---

## 4. General Solver Rules

### 4.1 Clearly Allowed

No reviewer approval needed:

1. Load pretrained backbone / general-purpose weights (timm, HF, similar) **inside** the submission script (§3.3). Starting point ≠ bringing your own already fine-tuned model.
2. Real training/fine-tuning entirely inside the submission script, including HPO (§1.1). Learning at submission time = correct.
3. Inference-time techniques that work in real production with **one sample** (or one small batch, e.g. BN) at a time, **zero** visibility into the rest of the test set:
   - TTA (per-sample) — fine
   - Batch-norm recalculated at batch level — fine
4. Sharing solution/approach **publicly** on Discord `#general` — encouraged (esp. challenge creators).

### 4.2 Clearly Not Allowed

Immediate rejection territory:

1. Pure algorithmic / rule-based solutions with **no** real training/fine-tuning — even if they “work.”
2. Train/fine-tune **outside** the script, host weights (e.g. HF), load your own already-fine-tuned weights back. Only general-purpose backbone weights allowed.
3. **External datasets** — any model, any challenge. Only challenge data (+ pretrained backbones per §4.1).
4. **Private** sharing of solutions/code/approaches between solvers on the same challenge (≠ public `#general`, which is allowed).
5. Using the **test set** beyond genuine one-sample / one-batch inference. Banned even without test labels:
   - Pseudo-labelling
   - Reweighting train samples using test data
   - Test-time adaptation
   - Calibrating outputs using full test-set distribution  
   Rationale: realism / production inference; must not adapt to this one test set.
6. Generating **synthetic data** and training on it — not allowed here (measure what you earn from given data only).

### 4.3 Grey-Area Techniques

Use only if you accept rejection risk.

**General self-test:** Strip the ML model out entirely. If the solution still basically works → not compliant. The model must do the learning (§1.1), not the solver working around it.

Challenge rules may explicitly allow e.g. n-grams / Markov as a starting point → you won’t be rejected *purely* for using them, but reviewers can still reject for over-reliance / non-generalization.

#### 4.3.1 Regex (esp. NLP)

| Use | Status |
|-----|--------|
| Regex itself | Not a violation |
| Cleaning data; deterministic extracts (e.g. numbers) | Generally fine |
| Over-reliance / exploiting generation process / overfitting distribution instead of ML | Reject |

Use as little as possible; prefer ML robustness.

#### 4.3.2 TF-IDF, N-grams, Markov Chains, Similar Stats

- Grey for same reason: tied to **this dataset’s distribution**, not generalizable like a trained model.
- Reviewer discretion per competition; can allow for one participant and not another.
- Depends mainly on: (1) quality/novelty of solution, (2) how much it relies on the technique.
- Use only if okay with that.

### 4.4 Challenge-Specific Rules

**Real overrides (follow them):**

- Restricted / required model types.
- Challenge-specific model-size limits.
- Rare case: challenge max runtime under 1 hour (e.g. 30 min) → follow it.

**Ignore as leftover boilerplate (not real overrides):**

- “no internet access”
- “decoding tricks are fine”
- “fully rule-based solutions are okay”  
These contradict §§3.3 and 4.2.

If unsure real restriction vs boilerplate → **ask a reviewer** before submit (same as §3.2). Do not gamble on appeal.

**Note in source:** §4.4 also states overall max runtime stays capped at 1 hour with pointer to §3.5; §3.5 states expected max **1.5 hours** (+ grace). Prefer §3.5 numbers for the default budget; treat challenge-specific shorter caps as binding when present.

### 4.5 Appealing a Rejection

1. Tag/inform the reviewer on your private Discord channel.
2. Explain in detail why rejection was wrong.
3. Timing: review/rejection happens **after** competition ends only.
4. Credits (local/global) are **never** refunded — whether rejection stands or appeal wins. Consumed at use time.

### 4.6 Accounts and Team Solving

| Rule | Detail |
|------|--------|
| Accounts | One account per person; multi-account = immediate ban, no warning |
| Team solving | Not on platform; not allowed even informally |
| Public sharing on `#general` | Allowed and encouraged (§4.1) |
| Private 1:1 collaboration | Not allowed |

---

## 5. Domain-Specific Guidelines

### 5.1 NLP / Seq-to-Seq Challenges

- Re-read §4.3 (regex, TF-IDF) before starting — highest temptation area.
- Many datasets are **synthetic** → may have exploitable structure.
- **Noticing** structure is fine; **gaming it directly** (regex / prebuilt templates) is not.
- Pattern/shortcut must be discovered and used **by the model during training** (§1.1), not hardcoded outside it.

### 5.2 Computer Vision and Object Detection Challenges

Synthetic-data exploits apply; tabular models love them.

| Technique | Status |
|-----------|--------|
| Tabular models directly on raw pixels | **Immediate reject** |
| Hand-engineered image features (histograms, edge counts) **into a real DL CV model** alongside the image | Fine |
| Same features **only** into a tabular model, no real image model | Grey (fragile; may reject) |
| Required approach | Real CV model from scratch or fine-tune (CNNs, ViTs, etc.) |
| Tabular on embeddings from pretrained backbone | Grey (submittable; may reject if non-robust) |

### 5.3 RAG and Retrieval Challenges

Second-highest violation domain after CV.

| Component | Requirement |
|-----------|-------------|
| Ranking / generation on top of retrieval | Must be **genuinely trained** (usually deep learning ranking model) |
| Hand-engineered features + classic off-the-shelf ranker | **Not enough** |
| Frozen off-the-shelf embedding model for **retrieval** | Fine (like pretrained backbone) |
| Inference-only overall | **Not allowed** |
| Fine-tuning the retriever | Recommended, not strictly required; fully frozen end-to-end pipeline won’t fly |

### 5.4 Fine-tuning Challenges

Must actually **fine-tune** a pretrained backbone.

| Approach | Status |
|----------|--------|
| Frozen embeddings → tabular head only | Grey (like §5.2); may reject for dodging fine-tuning / non-robust |
| Bio/chem categorized as **NLP (§5.1)** | Tabular on frozen embeddings = fine (no grey) |
| Bio/chem categorized as **Fine-tuning** | Leniency gone: must genuinely fine-tune domain model (e.g. ChemBERT); DL mandatory; XGBoost/LightGBM alone not tolerated |
| Hand-engineered features as extra inputs to model head | Fine |
| Ensemble fine-tuned model with something else | Grey; accepted only if performance/scores genuinely strong (reviewer call) |

### 5.5 From-Scratch Challenges

- Train from scratch on provided data only.
- **Any** pretrained model (embeddings, distillation, etc.) = **not allowed**.
- Same time budget as elsewhere (§3.5); no extension for from-scratch.
- Tokenizers / BPE / similar auxiliaries:
  - Not automatically banned.
  - Follow challenge description if it specifies.
  - If silent: usually OK by default (not counted as “pretrained models”).
  - Reviewers may still reject if they give unfair edge vs true from-scratch.
  - For certainty: ask reviewer, or use nothing pretrained.

### 5.6 Biology, Chemistry, and Other Domains Without Their Own Section

- No dedicated subsection yet → usually bucketed under best-fit category (often NLP §5.1 or Fine-tuning §5.4).
- Regex / TF-IDF-style / tabular techniques (grey in §4.3 or flagged in §5.2) tend to be **relatively more rule-compliant** here — molecular formulas/sequences make them a natural fit, not pure shortcut.
- Still reviewer judgment; not auto-accepted.
- **Exception:** if categorized as Fine-tuning → §5.4 stricter rules apply (genuine domain-model fine-tuning required).

---

## Appendix A — Hard Constraints Summary

```yaml
must:
  - training_or_finetuning_inside_submission_script: true
  - end_to_end_from_raw_data_each_run: true
  - libraries: kaggle_docker_image_only
  - data: challenge_data_only
  - pretrained_weights: general_purpose_backbones_only  # timm / HF / similar, loaded in-script
  - inference: one_sample_or_one_batch_visibility_only
  - accounts: one_per_person
  - team_solving: false

must_not:
  - inference_only_solutions
  - hardcode_dataset_insights
  - load_own_already_finetuned_weights
  - external_datasets
  - synthetic_training_data
  - private_solution_sharing
  - install_from_github_or_custom_libs
  - cache_artifacts_across_submissions
  - test_set_adaptation_techniques  # pseudo-label, TTA over full set, calibrate on full test dist, etc.
  - multi_accounts

time:
  expected_max_hours: 1.5
  grace_max_extra_minutes: 30
  recommended_train_stop_minutes: [50, 55]

compute:
  gpu: Nvidia A10G

credits:
  local_per_competition: 6
  local_refill: "+1 every 4 hours"
  global_daily_guide: "15-25 (variable)"
  global_refill: "per credit, 24h after use"
  csv_check_upload_costs_credit: false
  script_submission_costs_credit: true
```

---

## Appendix B — Grey-Area Self-Test

```
IF remove_ml_model(solution) still_basically_works:
  → NON_COMPLIANT or GREY_HIGH_RISK
ELSE IF model_genuinely_learns_insight_at_submission_time:
  → COMPLIANT_DIRECTION
```

Regex / TF-IDF / n-grams / Markov / tabular-on-embeddings: proceed only with acceptance of rejection risk unless challenge category explicitly relaxes (e.g. bio/chem as NLP).

## Submission format 
Solution submission instructions
Upload one Python script or notebook and one sample submission CSV. You do not enter paths in the UI—the platform supplies them when your code runs.

Runtime command

python3 solution.py <public_dir> <submission_out>
Starter pattern

Copy starter
import sys
from pathlib import Path
import pandas as pd

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")

# Images, if present:
 image_path = public_dir / "train" / f"{image_id}.png"

# Train and predict...
submission_out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_out, index=False)
Old

Path("dataset/public")
Path("working/submission.csv")
New

Path(sys.argv[1])
Path(sys.argv[2])
