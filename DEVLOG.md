# DEVLOG — gta6-trophy-rarity-forecast-01

Entry: `main.py` (Python 3.13, standard library only, no network). It reads
every `<id>.in` in the input dir and writes `<id>.out` with five quantiles
(5/25/50/75/95%) of each trophy's Exophase earn rate at the case's target.
Scoring is pinball loss on the logit scale; a prize requires beating the best
house entry on the final case (persistence starter, a fitted-curve baseline,
and a house AI forecast). Final case: history launch → 2026-11-21T11:30Z,
target 2026-11-26T00:00Z, horizon 6,510 min (4.52 days).

## Design

The recorded history is a step function (Exophase refreshes a game's counts
hours apart; consecutive reads are identical), so the program first dedupes
reads into distinct states. Two forces move a rate: player progress (earners
grow) and dilution (tracked population grows; new trackers hold fewer
trophies). On the logit scale they separate: with earner growth `a` and
population growth `b`, `d/dt logit(rate) = (a - b) / (1 - rate)`.

**Multi-state histories** (the final case): a weighted ensemble of four
candidates fit at runtime from the case's own history —

1. persistence (last logit carried forward);
2. a replica of the house fitted-curve baseline (OLS of logit rate on
   ln(1 + hours since launch) over the last 24 h of reads, read at target);
3. a damped log-time trend (OLS over recent distinct states, anchored at the
   last logit, extrapolation shrunk by span / (span + horizon));
4. a decomposition model that integrates `(a_x - b_x) / (1 - rate)` in
   log-time `x = ln(1 + hours)`: per-trophy earner growth minus
   tracked-population growth (from a saturating population fit — logistic or
   power law, chosen by a leave-the-future-out test), with shrinkage toward
   the cross-trophy median, decay in x, and a saturation ceiling brake.

Candidate weights come from a rolling-origin backtest inside the history
(fit on prefixes, score against the last state, origins weighted by recency —
early origins are dominated by the launch transient and misjudge candidates).
The quantile spread is calibrated from the ensemble's backtest residuals,
grown to the case horizon with a sub-square-root exponent (much of the
residual is cross-trophy dispersion, which does not grow like a random walk)
and blended with the random-walk spread `0.5 * sqrt(days)`.

**Single-state (frozen) histories** (the practice cases): no trend exists, so
a dilution-aware prior built in earners space, calibrated from the first-
refresh aggregates published in the kit's own `competition.yaml`: tracked
638 → 3,868 (×6.06), total earners 943 → 4,276, of which the tutorial trophy
is ~3,868 (100% of tracked) — so non-tutorial earners grew only 305 → ~408
(+34%) while the denominator sextupled. The prior is a two-branch mixture:
no-refresh (the frozen state itself, near-exact) vs refresh (tracked jumps
several-fold; tutorial stays ~100%; the other trophies' earners grow +34% in
total, allocated in proportion to current earners with a floor, tilted toward
early-game/high-rate trophies; rates land at roughly 0.15–0.3× their frozen
values), with the refresh probability from a hazard on the freeze's age and
the branch spread widening for few-earner trophies. Mixture quantiles are
inverted exactly by bisection.

Nulls, 0% and 100% rates are handled throughout (clamp to [0.01, 99.99]
before logits; last-non-null carry; never-observed trophies get a wide
prior; the ~100% tutorial trophy gets a tight spread at ~100%).

## Experiments and numbers

### Practice cases p00 / p01 (frozen histories)

Outputs are valid, deterministic (two runs byte-identical), 2.8 KB each,
~0.09 s for both cases together. Our medians vs persistence's (which carries
the last rate forward): for typical trophies our median logit is **−1.43 to
−1.55** below persistence's — e.g. id 4 (last rate 1.10%): our
q05/q50/q95 = 0.072 / 0.264 / 1.071 vs persistence 0.897 / 1.100 / 1.348;
id 20 (2.04%): ours 0.486 median vs 2.040. The tutorial trophy stays at
99.97–99.99 across all quantiles.

The true practice answers (the first refreshed state, per-trophy) are not
published, so the frozen prior was stress-tested against 40 synthetic
refreshes drawn from a *different* generative rule than the forecaster's
(sqrt-earners allocation, lognormal per-trophy noise, population jump drawn
×4.5–7.5): **our mean pinball 0.197–0.198 (range 0.18–0.25) vs persistence
1.347** on the same draws (persistence's real practice score is 1.51).
Context from the live practice board: the head below 0.05 is answer-copies
(the refreshed state was public on launch day; medians equal the answers,
typical miss 0.00) — not a genuine forecast and worthless for the final.
The honest frontier is ~0.08–0.4 (board median 0.424, rank ~100 ≈ 0.081);
our stress-test estimate sits inside that band, with the caveat that the
synthetic answers are model-based.

### Synthetic final-like cases

A cohort-based generator (launch players + later syncers with head starts,
per-trophy required play-hours and ever-earner shares, Exophase-style
refreshes every 1.5–6 h, reads every 10 min) calibrated so its ~26 h
snapshot roughly matches the real live page (top rates ≈ 1.5–2.8% vs real
1.89%, tutorial ≈ 100%). 11 variants (population growth, play pace, refresh
cadence varied), each cut at the real final checkpoint with the real
6,510-minute horizon, scored against the generated answering read:

| forecaster | mean loss | median loss |
|---|---|---|
| **this entry** | **0.4583** | **0.4624** |
| house fitted-curve replica | 0.4631 | 0.4665 |
| persistence | 0.6815 | 0.6826 |

The entry beats persistence on 11/11 variants and the curve replica on
7/11; 90% coverage 0.85–0.96 on the three named variants. Runtime
96–126 ms per final-size case (358 reads) — far under the 8 s cap.

### Internal backtest harness (prefix → held-out last state)

33 cuts (11 histories × checkpoints at ~30 h / 42 h / 54 h, horizons
~6–30 h), each forecaster fit on the prefix only:

| forecaster | mean loss | median loss |
|---|---|---|
| this entry | 0.1660 | 0.1378 |
| persistence | 0.2086 | 0.1818 |
| house fitted-curve replica | 0.1559 | 0.1406 |

At these short in-history horizons the pure curve replica is marginally
better on mean (the ensemble dilutes it); the ensemble is built for the
4.52-day final horizon, where the synthetic final eval above shows parity
or better, and on the real history its weights are re-derived from the real
in-history evidence.

### Design iterations that mattered

- **Clock-time decomposition failed its backtests**: integrating `(a − b)`
  in clock time overshot at every observable horizon (per-origin losses
  0.51–0.87 vs curve's 0.11–0.43) because launch-phase earner growth does
  not persist in clock time. Rewriting it in log-time made it competitive
  at origins (best at several) and kept its structural edge (it can see
  earners growing underneath a dilution-falling rate).
- **Recency-weighted backtest origins**: unweighted origins are dominated
  by the launch transient and scored the decomposition 0 weight even when
  its final-horizon medians were the most accurate.
- **Dead end — per-trophy spread from local fit noise**: on a step-function
  series, flat steps read as certainty; coverage collapsed to 0.64–0.76.
  Removed; the spread is global (with tutorial / few-earner adjustments).

### Edge cases verified

Valid output for: nulls in the last read / scattered nulls / a trophy null
in every read; a two-state history; a truncated few-state history; all-zero
rates except the tutorial. A deliberately slowed-progress, fast-population
variant still beat persistence (0.387 vs 0.520) and stayed near the curve
replica (0.346) — though within this generator family the final window is
progress-dominated (population growth saturates while progress accumulates),
which is also the realistic expectation for week one.

## Limitations

- The frozen prior is validated against published aggregates and synthetic
  refreshes, not per-trophy practice answers (not public).
- The synthetic generator is one plausible envelope, not ground truth; the
  program's main defence is that candidate weights and spread are
  re-derived from each case's own history at runtime.
- The house AI forecast is an unknown quantity; the bar used here is the
  better of the two documented baselines.
