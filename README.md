# Business Entity Resolution Challenge (Amazon ML)

Match every Source 1 business record to all of its duplicates in Source 2 and Source 3
(US, India, France). Metric: mean F0.5 per S1 record (precision weighted over recall).

## Pipeline

1. **Normalization (v6)** - lowercasing, legal-form mapping (private->pvt, limited->ltd, ...),
   address cleanup, house-number parsing, per-country rules.
2. **Blocking** - TF-IDF name, address, combined and reverse blockers, state buckets,
   cap 30 candidates per source per S1.
3. **Stage A** - LightGBM pair model on local features (name/address similarity, numbers,
   legal form, token overlap), 5-fold GroupKFold by S1 plus a locked holdout.
4. **Stage B** - LightGBM context model on top of Stage A scores: group view, rank and margin
   inside each S1, rival competition, number delta, legal conflict, sibling words,
   name frequency, phonetic skeleton.
5. **Decision rule** - one-to-one ownership, then per S1 pick the list length that
   maximizes expected F0.5 (strictness v).
6. **Stage C (experimental, branch `stage-c`)** - re-ranker on Stage B scores,
   3-seed ensemble, blend weight, per-country strictness.

## Results (local validation, F0.5)

| Step | Dev | Holdout | India | US |
|---|---|---|---|---|
| Stage A v6 | 0.9682 | 0.9699 | 0.9575 | 0.9758 |
| Stage B v1 + expected-F0.5 rule | 0.9784 | 0.9791 | 0.9701 | 0.9841 |

Stage B gave +0.010 over Stage A.

## Checks and findings

- **Blocking ceiling:** 98.47% of true pairs reach the candidate list; after pruning 98.25%.
  Blocking is the hard limit (max possible score about 0.994).
- **Sibling -> singleton:** singleton rate is flat at 5.58% with or without S1 siblings. No signal.
- **Twin rule:** S2/S3 records with identical name + address always share one owner
  (precision 1.0, never orphans), but Stage B already handled them (gain +0.0000).
- **Orphans:** about 26% of S2/S3 records in train belong to no S1.
- **Loosening pruning** (p>=0.003, top 5) recovers only about +0.0005.

## How to run




Test scoring can be split across notebooks with `BER_PART` / `BER_NPARTS` environment variables.
Paths are set with `BER_DATA_DIR`, `BER_WORK_DIR`, `BER_OUT_DIR`.

## Lessons learned

- Blocking recall caps everything; fix it first.
- A context model plus an expected-F0.5 decision rule gave most of the gain.
- Training on only 5% of S1s hides competition between S1s; full-train scoring is the next big step.
- On Kaggle: use Save & Run All (commit) for long jobs, back up models as datasets,
  and split heavy scoring into parallel notebooks to stay under the 12-hour limit.

## Next ideas (not finished)

Record-level model with full competition features, new exact-name and phonetic blockers,
cutting lists by predicted group size, and better India transliteration handling.

---
Built over 3 days of grinding on Kaggle.
