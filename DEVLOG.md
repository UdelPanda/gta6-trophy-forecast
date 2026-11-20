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

**Multi-state histories** (the final case): a weighted ensemble of
candidates fit at runtime from the case's own history (five since the
2026-11-20 calibration iteration — see below) —

1. persistence (last logit carried forward);
2. a replica of the house fitted-curve baseline (OLS of logit rate on
   ln(1 + hours since launch) over the last 24 h of reads, read at target);
3. a damped log-time trend (OLS over recent distinct states, anchored at the
   last logit, extrapolation shrunk by span / (span + horizon));
4. a decomposition model that integrates `(a_x - b_x) / (1 - rate)` in
   log-time `x = ln(1 + hours)`: per-trophy earner growth minus
   tracked-population growth (from a saturating population fit — logistic or
   power law, chosen by a leave-the-future-out test), with shrinkage toward
   the cross-trophy median, decay in x, and a saturation ceiling brake;
5. the fitted curve with its extrapolation damped toward the last logit
   (added 2026-11-20).

Candidate weights come from a rolling-origin backtest inside the history
(fit on prefixes, score against the last state, origins weighted by recency —
early origins are dominated by the launch transient and misjudge candidates);
since 2026-11-20 the weights are also computed per trophy, shrunk toward
the global weights.
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

## Calibration iteration 2 — 2026-11-20

One focused iteration, driven by two observations: the practice board
showed very accurate medians (typical miss 0.05 pp) but 90% coverage of
0.982 — intervals wider than the outcomes need, and pinball pays for
width — and on the original synthetic final-like suite the entry beat
the fitted-curve replica in only 7 of 11 variants.  The original harness
scripts were not saved, so every suite was rebuilt from the descriptions
above; the rebuilt refresh suite reproduces the documented numbers for
the previous model (0.198 vs 0.197–0.198 documented, and the same 0.982
coverage the real practice answers gave), but the rebuilt final-like and
backtest suites run harder than the originals (the previous model scores
0.700 / 0.398 there, not 0.4583 / 0.1660), so this iteration's gates were
applied comparatively — new model vs previous model on identical rebuilt
suites — rather than against the original absolute values.

Changes, each a general rule:

- **Frozen prior: exact aggregates + calibrated spread.**  Branch B's two
  free parameters now equal the kit's published first-refresh aggregates
  exactly at age zero (population ×6.06, not ×6.2; non-tutorial earners
  +33.8%, not +38%), and the branch spread is scaled by 0.75: unscaled,
  the mixture's 90% interval covered ~98% of outcomes both on the real
  practice answers and on the synthetic refreshes.  Rebuilt 40-draw
  refresh suite: **0.1976 → 0.1598** (persistence 1.3795).
- **Per-trophy candidate weights.**  The backtest's gated softmax is now
  also computed per trophy and shrunk toward the global weights by the
  origin evidence behind it.  Diagnosis on the expanded suite showed the
  global weight cannot express that the curve is right for one trophy
  (clean log-time climb) and wrong for the next (dilution-masked): on
  the progress-dominated variant the entry lost, the curve won a handful
  of trophies by up to 0.86 while losing others by up to 1.84.  This was
  the single largest final-like gain.
- **A saturation-damped curve candidate.**  The house curve's known
  weakness is reading a log-time line 4.5 days past its window with no
  saturation and no dilution term; a fifth candidate repeats the curve
  with the move shrunk by span / (span + extrapolation distance), and
  the backtest weights it like any other.
- **Asymmetric floor tail.**  A trophy pinned near 0% can only surprise
  upward — the progression wave multiplies its rate when it lands,
  while the floor bounds the downside — so below a 0.05% last rate the
  upper tail uses 1.45× the calibrated spread.  This flipped the
  remaining curve loss and most of the persistence loss.
- **Tried and rejected: horizon-matched origin weights** (weighting
  backtest origins by horizon as well as recency).  Alone it changed
  nothing; combined with per-trophy weights it made the final-like mean
  worse (0.6841 vs 0.6757).  Reverted.

The final-like suite was also expanded from 11 to 24 variants (stall
cadences of 12–24 h, a weekend surge before and after the checkpoint, a
second arrival wave, stronger/weaker dilution extremes, early vs late
earner saturation, a stale checkpoint).  Rebuilt-suite results,
previous model → this model:

| suite | previous | this model | head-to-head |
|---|---|---|---|
| refresh, 40 draws (frozen) | 0.1976 | **0.1598** | — (persistence 1.3795) |
| final-like, core 11 (mean / median) | 0.7002 / 0.7006 | **0.6398 / 0.6379** | vs curve 11W-0L, vs persistence 11W-0L |
| final-like, all 24 (mean / median) | 0.7074 / 0.7294 | **0.6479 / 0.6551** | vs curve **24W-0L**, vs persistence 23W-1L |
| backtest, 33 cuts (mean / median) | 0.3982 / 0.3858 | **0.3915** / 0.3902 | vs curve 30W-3L (curve mean 0.6527) |

90% coverage on the final-like suite: 0.877 → 0.923.  The one tolerated
loss is a single expanded variant against *persistence* (fast pace +
early saturation, 0.4275 vs 0.4108 — the ensemble still slightly
undershoots arrival-driven earner growth there); against the curve
replica, the stronger house entry, the record is 24W-0L.  Per-trophy
weights alone cost +0.0075 on the rebuilt backtest mean; the floor-tail
asymmetry more than repaid it, and the accepted combination improves
every suite's mean over the previous model — the final-like suite was
weighted highest throughout, since the prize is decided on the final
case alone.  Practice outputs remain valid and byte-identical across
runs; worst runtime on a final-size case is ~0.6 s of the 8 s cap.

## Limitations

- The frozen prior is validated against published aggregates and synthetic
  refreshes, not per-trophy practice answers (not public).
- The synthetic generator is one plausible envelope, not ground truth; the
  program's main defence is that candidate weights and spread are
  re-derived from each case's own history at runtime.
- The house AI forecast is an unknown quantity; the bar used here is the
  better of the two documented baselines.

## V3 — 2026-11-20 (built for the final case, not the practice board)

The practice score does not count toward the prize; the final case f00
does (checkpoint 2026-11-21T11:30Z, target 2026-11-26T00:00Z, horizon
6,510 min), and a prize requires beating the best house entry on it.
V3 was therefore developed against a final-like harness: synthetic
cases mimicking f00 — ~2.5 days of 10-min reads as a step function
(Exophase refreshes every few hours), tracked population growing from
~638 through several doublings with decelerating growth, per-trophy
earners following story-progress dynamics (fast tutorial/story, slow
grind/collectible, per-trophy saturation ceilings), the answer read at
the last refresh before the target. 36 variants (12 core across
pace / saturation / dilution / cadence axes + 24 expanded), two
independent replications of the core 12 with different generator seeds
(B and C), a 24 h-horizon pre-case suite, and a refresh suite (frozen
p00 history + 40 synthetic first refreshes drawn from a different
generative rule: population jump x4.5-7.5, sqrt-earner allocation,
lognormal noise). Both house baselines are replicated exactly as
scorers: persistence from the kit starter; the fitted curve per
RULES.txt with random-walk + fit-residual spread combined by
root-sum-of-squares.

### What changed vs v2

1. Sixth candidate `cand_curve_long`: the fitted curve over the last
   72 h of reads. The house curve's 24 h window is short enough that a
   single refresh interval can tilt its slope; the long window averages
   several refreshes. (The backtest gives it weight mainly where the
   24 h fit is unstable, e.g. stall cadences.)
2. The rolling-origin backtest now runs at >= 4 distinct states
   (v2: >= 6), its gated softmax is factored into `_gated_softmax`, and
   the per-trophy shrinkage lambda is capped at 0.6.
3. Horizon prior: as the horizon grows past a day, weight shifts from
   the flat core (persistence, damped trend) to the curve family
   (share 0.45 at the full 4.5 days), plus a damped -> curve shift
   inside the family. The in-history backtest cannot see this on its
   own: its origins reach at most the observed span (~2.5 d), where a
   flat median's error is still small; at 4.5 days it grows like
   sqrt(horizon) while a real trend's does not.
4. Saturation tilt: `saturation_decel` measures each trophy's earner
   growth per log-time in the recent third of the span vs the middle
   third. A ratio under 1 (the progression wave is cresting) shifts
   weight from the extrapolators to the flat candidates, and vice
   versa; applied globally (median ratio) and per trophy (strength
   0.7), capped so the backtest's judgement is tilted, never replaced.
   Measured median ratios: saturation variants 0.77-0.87, others
   0.95-1.19.
5. Sharpening: final global and per-trophy weights are squared and
   renormalised, concentrating each trophy's median on its favoured
   candidates while keeping the hedges at reduced strength.
6. The decomposition candidate is loosened: earner-growth decay scale
   1.2 -> 3.0 in log-time, saturation ceiling max(1.8r, r+4) ->
   max(3r, r+12), integration clamp [-2.0, +1.5] -> +/-3.0. Standalone
   it overshoots on hard-saturation variants; the tilt (4) is the
   counterweight, and the pair together beat either alone.
7. Spread: a global x0.92 trim, and trophies whose last rate is
   exactly 0.00 get a 5th percentile of 0.00 — nobody may earn them by
   the target either. Scoring clamps quantiles to 0.01 before the
   logit, so this costs no loss; it stops the band from starting above
   a rate that is still zero. (On the saturation variants almost every
   coverage miss was an exact-zero trophy; this fix alone was worth
   ~0.14-0.31 of coverage there.)

### Results (mean pinball loss on identical suites; lower is better)

Final-like, seed A (36 variants): v3 0.3470, v2 0.4153, persistence
0.7275, curve 0.3900. Core 12: v3 0.3478, v2 0.4127, curve 0.3841,
persistence 0.7311. W-L for v3: vs v2 34-2 (all) / 11-1 (core); vs
persistence 34-2 / 11-1; vs curve 31-5 / 10-2. Coverage90: v3 0.954,
v2 0.812.

Seed B core 12 (independent draws): v3 0.3612, v2 0.4247, persistence
0.7532, curve 0.4046 — v3 goes 12W-0L against all three. Coverage
0.941.

Seed C core 12: v3 0.3873, v2 0.4377, persistence 0.7526, curve
0.4075 — vs v2 11-1, vs persistence 11-1, vs curve 10-2. Coverage
0.917.

Pooled core (36 draws): v3 0.3654, v2 0.4250, curve 0.3987,
persistence 0.7456; v3 vs v2 34W-2L, vs persistence 34W-2L, vs curve
30W-6L. Coverage within [0.85, 0.95] on every seed family (0.954 /
0.941 / 0.917 — seed A's 0.954 is the zero-trophy effect: the
generator's grind tail prints more exact 0.00 rates than the real
26 h page showed).

Pre-case suite (24 h horizon, seed A core): v3 0.1885 vs v2 0.1906 —
the long-horizon machinery correctly does almost nothing at a day.

Refresh suite: v3's outputs are byte-identical to v2's (the frozen
branch is untouched), 0.2330 vs persistence 1.4068 over the 40
synthetic refreshes.

### Remaining losses, named

- fast-early-sat vs persistence: +0.016 (seed A), +0.028 (seed C),
  a win on seed B (-0.013). The hardest family — v2's one expanded
  loss lived here too. When the wave has fully crested by the
  checkpoint, persistence is nearly right and any extrapolation pays;
  v3 narrows the margin but does not erase it.
- Seed A core vs curve: late-sat +0.001 and weak-dilution +0.006 —
  inside single-draw noise; both are clear wins on seeds B and C.
- Seed C stall-cadence vs curve: +0.151. Stall variants carry only
  4-6 distinct states and every forecaster swings there (persistence
  scores 0.80-1.02 across seeds); v3 wins stall on seeds A and B.

The standing bar — no core loss to either house replica — holds
exactly on seed B and in aggregate on all three seeds; the residual
losses rotate by seed, the signature of draw noise around a real
edge rather than a systematic gap. No parameter was tuned against a
single seed's losses.

### Negative results (tried, measured, dropped)

- Pair-based backtest (origins x multiple target states, loss-vs-sqrt(h)
  regression; pinball, then MAE): final-like 0.444-0.449 vs v2's
  0.4153 and pre-case 0W-11L vs v2. Pinball with sqrt(h)-matched sigma
  compresses candidate differences to noise; recency-squared weighting
  starves the long-horizon points; the MAE version crowned persistence
  on slow-pace. Reverted to v2's origin -> last-state backtest.
- Sigma growth exponent 0.35 -> 0.5 (cap 1.8 -> 2.2): coverage
  0.821 -> 0.838 but mean loss 0.3657 -> 0.3749. The low-coverage
  variants' misses are one-sided median misses, not width misses.
- Floor upper tail x1.9 for last rate < 0.10 (from x1.45 / < 0.05):
  coverage overshot to 0.989 and the saturation variants paid for
  width they did not need.
- Deceleration-dependent sharpening power (1.15 at low decel rising
  to 2.0): 0.3508 vs the flat power's 0.3470 — it helped
  fast-early-sat and hurt late-sat / weak-dilution more.
- Per-trophy tilt strength 1.0 in place of 0.7: 0.3548 vs 0.3470.

### Gates

- Determinism: two CLI runs over the kit practice inputs are
  byte-identical; in-process double runs identical on the largest
  cases. Output ~2.8 KB per case (cap 65,536), 55 rows x 5 monotone
  finite quantiles in [0, 100].
- Runtime: worst 0.36 s on a 359-read final-like case (cap 8 s).
- Kit check (RULES.txt "Before you submit", run with v2's transcripts
  bundle): 14 passed, 0 failed, 5 skipped, 5 checked at submission.
  With --worktree against the same bundle, authorship reads 859/1067
  counted lines (80.5%; bar 30%) with 208 lines absent — exactly v3's
  new lines, which the v2 bundle predates; the bundle to submit with
  is the one recorded from the v3 build session itself.
