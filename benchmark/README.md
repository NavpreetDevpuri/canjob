# Benchmark: does the ranking actually work?

The challenge data ships without ground-truth labels, so this folder validates the
ranking method on an **independent, publicly labelled** dataset and reports the exact
competition metrics (NDCG@10, NDCG@50, MAP, P@10).

## Dataset

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
(test split): 1,759 `(resume, job_description)` pairs, each with a human label
**Good Fit / Potential Fit / No Fit** (graded relevance 2 / 1 / 0), across 71 jobs and
477 unique resumes. For each job we rank **the entire pool of 477 candidates** (the real
retrieval task) and check how high the recruiter's Good-Fit picks land. Resumes the
dataset did not judge for a job are treated as No-Fit, which is conservative: a genuinely
good but unjudged candidate we rank highly counts *against* us, so the true quality is at
least what is reported. Binary metrics (MAP, P@10) count only **Good Fit** as relevant;
NDCG is graded.

## What is tested

The **generalisable recall + fusion core** of CanJob, the same code paths used in the
product, with three lenses fused by the identical `canjob.featurize.rrf_merge`:

| lens | what it captures |
|---|---|
| `tfidf_word` (1–2 gram) | keyword overlap |
| `tfidf_char` (3–5 gram) | typo / morphology-robust overlap |
| `embeddings` (MiniLM) | meaning match |

The Redrob-specific layer (deterministic JD rules, honeypot detection, the
off-domain eligibility gate) is **not** exercised here because it is tied to the
Redrob candidate schema. That layer is validated directly on the challenge data:
it moves the off-domain "Content Writer | exploring GenAI" from **rank #1 to
#21,804 / 100,000**, with **0 honeypots in the top 100**.

## Run

```bash
pip install -r benchmark/requirements.txt    # adds torch+transformers (benchmark only)
python benchmark/run_benchmark.py            # downloads the dataset once, then scores
```

## Result (macro-averaged over 19 jobs, ranking all 477 candidates per job)

| method | NDCG@10 | NDCG@50 | MAP | P@10 | P@10 vs random |
|---|---|---|---|---|---|
| random floor | 0.049 | 0.084 | 0.061 | 0.047 | 1x |
| tfidf (word) | 0.205 | 0.235 | 0.137 | 0.189 | 4x |
| tfidf (char) | 0.150 | 0.215 | 0.124 | 0.126 | 3x |
| embeddings (MiniLM) | 0.210 | 0.257 | 0.150 | 0.211 | 5x |
| **ensemble (RRF, semantic-led)** | **0.219** | **0.267** | **0.156** | 0.200 | 4x |

**How to read the numbers** (all 0–1, higher = better; the `random floor` is the score
from shuffling, so the real skill is the *multiple* above it):

- **P@10** — fraction of our top 10 that are genuine "Good Fit": ~0.20 (≈ 2 of 10) vs
  ~0.047 by chance = **~4x more good fits at the top than random**.
- **NDCG@10 / NDCG@50** — is the top 10 / 50 in the right order, best-fit first? 1.0 = perfect.
- **MAP** — overall ordering quality across *all* the good-fit candidates.

**Why the absolute numbers are modest (and that's expected):** this is the hardest fair
test — rank the entire pool from scratch, count only the strict "Good Fit" as a win, and
treat any unjudged resume as a No-Fit (so a good candidate we surface but nobody labelled
counts against us). Several JDs are vague boilerplate with nothing concrete to match on,
and "fit" is a subjective recruiter call. Under those conservative rules a 4–5x lift over
chance is a strong, honest signal.

**Takeaways**

- Every learned lens beats the random floor by 3–5x on data we never saw or tuned on:
  the ranking method is sound, not overfit to the challenge JD.
- The **RRF ensemble is the best method overall** (top NDCG@10, NDCG@50 and MAP) — exactly
  what fusing three complementary lenses is supposed to deliver.
- The MiniLM semantic lens is the strongest single lens, validating that core choice.
- This tests only the generalisable recall core; the Redrob rules + eligibility gate (the
  real differentiator) is validated on the challenge data, where it drops the off-domain
  "Content Writer" from #1 to #21,804 / 100,000 with 0 honeypots in the top 100.

Numbers are reproducible with the command above and saved to `results.json`.
