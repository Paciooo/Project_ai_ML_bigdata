> Start here for the current small reports. Full local runs are listed in root CLAUDE.md.

# Canonical reports

Use these folders for the presentation and remote review:

| Evidence | Folder |
|---|---|
| Local A, next-close direction | `local_run_A_holdout_close/` |
| Local B, next-body direction | `local_run_B_holdout_body/` |
| Local C, four expanding test folds | `local_run_C_walkforward/` |
| NQ Timestral-5M, matched origins | `local_timestral_NQ_5M/` |
| Nearest-expiry VIX and coin controls | `coinflip_vix_front/` |
| Raw-input/model/output verification and corrected economics | `verification/` |

The superseded cloud A snapshot was removed from the working folder to
avoid mixing it with local A. Its historical files remain in Git history
(for example commit fee4f65). XGBoost outputs differed across those runs;
environment/source-state differences were not isolated to a unique cause.

The only retained old control is [synthetic_null_legacy/](synthetic_null_legacy/),
a clearly labeled GBM null summary. Do not treat it as a current timed run.

Raw VIX CSVs, multi-million-row NQ predictions and model weights remain local.
The verification report records their hashes and reconstructs labels and
predictions using those actual local inputs. It does not pretend that a
remote reader has access to files excluded from Git.
