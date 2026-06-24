# Scout log

Non-claim, exploratory probes (per `docs/COMPLETION_PREDICATES.md` Scout
procedure). A scout may peek, use one seed, or run unguarded. **It never enters a
verdict, an `ASSUMPTIONS.md`/`BATTERY.md` row, or a phase conclusion** — it informs
routing and hypotheses only. A scout's seed is **burned**: peeked/calibration data
cannot later be re-used as evidence. Promotion to a claim requires a fresh,
pre-registered confirmatory run on unburned seeds under the full review protocol.

| date | seed(s) | script | artifact | what it settled (routing only) | burned |
|---|---|---|---|---|---|
| 2026-06-24 | 777 | `scripts/predicate_sufficiency/calib_777.py` *(to commit before exp45 freeze — finding 5)* | `out/colored_dyck2-calib/` (`config.json`, `model.pt`, `cache.npz`) | colored-Dyck matching is grammar-forced → `matches_type` = `matches_color` = `matches_both` = 0.811 **identical**; the "next close matches the prefix top" predicate is **degenerate for composition**. Redirected exp45 from the `matches_*` suite to the **facet-value** suite (`phi_type0`/`phi_color0`/`phi_both00`). Also fixed plumbing references: estimator corr 0.98–0.999, per-prefix drift up to ~0.3 (→ pooled-mean only), ~180–200 determined-ctx prefixes per `t` per 250 seqs. | yes — excluded from exp45 claim seeds {801–804} |

## Pending (gate the exp45 freeze)

- **Commit `scripts/predicate_sufficiency/calib_777.py`** (the script that produced
  the seed-777 redirection above). Currently only the cached output
  (`out/colored_dyck2-calib/cache.npz`) is on disk; the redirecting numbers in
  `experiments/45-predicate-sufficiency-composition.md` are not reproducible until
  the generating script is committed beside the writeup.
- Seed **800** (the exp45 `--calibrate` threshold seed) will be logged here when it
  runs; it is also burned and excluded from {801–804}.
