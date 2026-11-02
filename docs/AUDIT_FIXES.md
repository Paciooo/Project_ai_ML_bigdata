# Audit fixes and local verification

Completed locally on 2 November 2026. The user did not need to send the local
files: this session accessed the actual NQ/VIX inputs, saved classifiers,
prediction Parquets and cached Timestral weights.

## Fixed

1. Entry timing: first observed clean minute open >= decision, inside the
   target interval/session, with strictly later exit. Invalid entries are
   rejected. One-minute economics with default positive latency is explicitly
   not evaluable. No fictional one-second fill is introduced.
2. Fold thresholds: materialized as prediction columns; missing metadata is
   an error. Existing thresholds were reselected on validation under corrected
   execution using saved models, then attached by fold id. Classifiers were
   not retrained and their probability/direction outputs remain identical.
3. Drawdown: end-of-day equity now includes its initial zero point.
4. Provenance: local A is canonical; B, C and NQ Timestral small outputs are
   available in `reports/`. The cloud A replica is labeled superseded.
5. Documentation: completed-run status and interpretation updated; causal
   bounce claims and claims of proved untradability removed from current docs.

## Verified against the actual local files

- NQ A/B/C labels rebuilt from minute history; every saved model reproduces
  its saved test probabilities. Classification accuracy is unchanged.
- NQ Timestral-5M was rerun from cached real weights on the original matched
  origins and its table was checked against the prediction files.
- VIX duplicates and source hashes checked; same-expiry labels rebuilt.
  Saved logistic/XGBoost outputs and all four Timestral directions reproduced.
- Detailed checks, model/input hashes and backup locations are in
  `reports/verification/local_verification.json` and each `rescore.json`.
- Original predictions/economics are backed up in local `audit_backups/`.
  Raw source files and model weights were not changed.

## Economic outcome

No learned strategy has a positive lower daily-P&L interval under both base and high costs.
Classification advantage is not the same as a profitable strategy. These
historical evaluations remain exploratory. Quotes, actual fills, validated
contract calendars/rolls and a fresh untouched holdout remain outside the
available evidence. Public reports contain small tables/charts/audit evidence,
not the large raw datasets or model weights.

## Tests

Final check: **70 passed** using the existing Windows/Python 3.12 environment.
Regressions cover post-decision fills, unavailable one-minute entries,
invalid timing rejection, per-fold thresholds, missing fold metadata,
UTC preservation for empty entries and initial losses in drawdown.

## Reproduction

```powershell
python -m pytest tests -q
python -u tools/repair_results.py
```

No full classifier training or model downloads are needed for this repair.
Do not rerun the experiment suite merely to obtain the already verified
presentation numbers. The revised presentation remains at most ten slides.
