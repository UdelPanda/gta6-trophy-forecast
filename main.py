"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads a case (the house's recording of Exophase's GTA VI PS5 trophy page up to a
checkpoint) and writes, for every trophy, five quantiles of its earn rate at the
case's target, scored by pinball loss on the logit scale.

How it forecasts
----------------
The recorded history is a step function: Exophase refreshes a game's counts only
every few hours, so consecutive reads are usually identical.  The program first
dedupes the reads into distinct states.  Two forces move a rate between states:
player progress (the trophy's earners grow) and dilution (the tracked population
grows, and newly tracked players hold fewer trophies).  On the logit scale the
two forces separate cleanly: if earners grow at relative rate a and the tracked
population at relative rate b, then d/dt logit(rate) = (a - b) / (1 - rate).

With two or more distinct states the program fits, at run time, from the case's
own history:

  * a saturating model of the tracked population (logistic or power law,
    chosen by a leave-the-future-out test on the population series itself);
  * four candidate forecasters per trophy: persistence, a replica of the house
    fitted-curve baseline (logit rate on ln(1 + hours since launch) over the
    last 24 h of reads), a damped log-time trend, and a decomposition model
    that integrates (a_x - b_x) / (1 - rate) in log-time (earner growth minus
    tracked-population growth, self-damping, with a saturation brake);
  * a rolling-origin backtest inside the history (fit on a prefix, score
    against the last state) that weights the candidates and calibrates the
    quantile spread from the ensemble's own logit residuals.

With a single distinct state (a frozen history, like the practice cases) no
trend exists to fit, so it falls back to a dilution-aware prior: a refresh is
likely before the target, the tracked population jumps when it lands, and each
trophy's rate is pulled toward the incoming cohort's level -- high-rate
trophies (earned early, also by newcomers) fall least, rare ones most -- with a
wide, honest spread from an explicit two-branch mixture (refresh / no refresh).

Nulls, 0% and 100% rates are handled throughout: rates are clamped to
[0.01, 99.99] before any logit, a trophy never observed gets a wide prior, and
the tutorial trophy at ~100% is forecast to stay there with a tight spread.

The program is deterministic (no randomness, no clock, sorted file order),
uses only the Python 3.13 standard library, and runs in well under a second
per case.  Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand
in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader runs `python3 -P -s /bot/main.py`, which keeps this file's
# directory off sys.path; put it back first in case a module ships beside it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(pct):
    """Logit of a rate in percent, the rate clamped to [0.01, 99.99] first."""
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg):
    """Inverse logit back to percent, overflow-safe."""
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residuals)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, [y - my for y in ys]
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    return a, b, [y - (a + b * x) for x, y in zip(xs, ys)]


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


# ---------------------------------------------------------------------------
# Case parsing: times, distinct states, per-trophy series
# ---------------------------------------------------------------------------


def parse_time(text):
    """ISO 8601 timestamp -> epoch seconds (UTC)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def launch_epoch(case):
    """Hours are measured from 00:00Z on the game's release date."""
    rel = (case.get("game") or {}).get("release_date")
    if rel:
        try:
            dt = datetime.fromisoformat(rel).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    hist = case.get("history") or []
    return parse_time(hist[0]["at"]) if hist else 0.0


class State:
    """One distinct observed page: a maximal run of identical reads."""

    __slots__ = ("t", "n", "earned", "rates", "end_idx")

    def __init__(self, t, n, earned, rates, end_idx):
        self.t = t            # hours since launch (of the run's first read)
        self.n = n            # players tracked (or None)
        self.earned = earned  # total trophies earned (or None)
        self.rates = rates    # per-trophy earn rate in percent (or None)
        self.end_idx = end_idx  # index of the run's last read in the history


def build_states(case, epoch):
    states = []
    prev_key = None
    for idx, rd in enumerate(case.get("history") or []):
        rates = list(rd.get("rates") or [])
        key = (rd.get("players_tracked"), rd.get("trophies_earned"), tuple(rates))
        t = (parse_time(rd["at"]) - epoch) / 3600.0
        if states and key == prev_key:
            states[-1].end_idx = idx
            continue
        states.append(State(t, rd.get("players_tracked"), rd.get("trophies_earned"), rates, idx))
        prev_key = key
    return states


def rate_at(state, i):
    if i < len(state.rates):
        return state.rates[i]
    return None


def earners_at(state, i):
    r = rate_at(state, i)
    if r is None or not state.n:
        return None
    return r * state.n / 100.0


# ---------------------------------------------------------------------------
# Tracked-population model: saturating growth, fit on the distinct states
# ---------------------------------------------------------------------------


def _softplus(z):
    if z > 40.0:
        return z
    if z < -40.0:
        return 0.0
    return math.log1p(math.exp(z))


def fit_power(pts):
    xs = [math.log1p(max(t, -0.9)) for t, _ in pts]
    ys = [math.log(n) for _, n in pts]
    a, b, _ = ols(xs, ys)
    return {"kind": "power", "a": a, "b": b}


def fit_logistic(pts):
    n_last = pts[-1][1]
    best = None
    for f in (1.15, 1.3, 1.5, 1.8, 2.2, 2.7, 3.5, 5.0, 8.0, 13.0, 21.0):
        cap = n_last * f
        if any(n >= cap for _, n in pts):
            continue
        xs = [t for t, _ in pts]
        ys = [math.log(n / (cap - n)) for _, n in pts]
        a, b, _ = ols(xs, ys)
        if b <= 0.0:
            continue
        sse = 0.0
        for t, n in pts:
            pred = math.log(cap) - _softplus(-(a + b * t))
            sse += (math.log(n) - pred) ** 2
        if best is None or sse < best[0] * 0.97:
            best = (sse, {"kind": "logistic", "a": a, "b": b, "k": cap})
    return best[1] if best else None


def pop_predict(pop, t):
    if pop["kind"] == "power":
        return math.exp(pop["a"]) * (1.0 + max(t, -0.9)) ** pop["b"]
    return pop["k"] / (1.0 + math.exp(-(pop["a"] + pop["b"] * t)))


def pop_log_deriv(pop, t):
    """d/dt ln N(t) under the fitted population model (per hour)."""
    if pop["kind"] == "power":
        return pop["b"] / (1.0 + max(t, -0.9))
    return pop["b"] * (1.0 - pop_predict(pop, t) / pop["k"])


def fit_population(pts):
    """pts: [(t_hours, players_tracked)] with at least 2 points, N > 0."""
    if len(pts) < 2:
        return None
    power = fit_power(pts)
    logistic = fit_logistic(pts) if len(pts) >= 3 else None
    if logistic is None:
        return power
    if len(pts) >= 5:
        m = max(1, len(pts) // 5)
        train, (t_last, n_last) = pts[:-m], pts[-1]
        err_p = abs(math.log(pop_predict(fit_power(train), t_last) / n_last))
        log_tr = fit_logistic(train)
        err_l = abs(math.log(pop_predict(log_tr, t_last) / n_last)) if log_tr else 1e9
        return logistic if err_l < err_p else power
    # Few points: compare in-sample fit in ln N.
    def sse(model):
        return sum((math.log(n) - math.log(pop_predict(model, t))) ** 2 for t, n in pts)
    return logistic if sse(logistic) < sse(power) else power


# ---------------------------------------------------------------------------
# Candidate forecasters.  Each takes a Ctx and returns a list of median
# logits (one per trophy) or None where it cannot speak for a trophy.
# ---------------------------------------------------------------------------


class Ctx:
    def __init__(self, case, epoch, states, reads, t_target):
        self.case = case
        self.epoch = epoch
        self.states = states            # distinct states (prefix, for backtests)
        self.reads = reads              # raw reads of the prefix [(t_hours, rates)]
        self.t_target = t_target        # hours since launch
        self.ntr = len(case["trophies"])
        self.t_last = states[-1].t if states else 0.0
        self.horizon_h = max(0.0, t_target - self.t_last)
        # Last observed rate / logit / earners per trophy.
        self.last_rate = []
        self.last_logit = []
        self.last_earners = []
        for i in range(self.ntr):
            r = None
            for st in reversed(states):
                r = rate_at(st, i)
                if r is not None:
                    break
            self.last_rate.append(r)
            self.last_logit.append(logit_pct(r) if r is not None else None)
            e = None
            for st in reversed(states):
                e = earners_at(st, i)
                if e is not None:
                    break
            self.last_earners.append(e)
        # Population fit on this prefix.
        pts = [(st.t, st.n) for st in states if st.n]
        self.pop = fit_population(pts) if len(pts) >= 2 else None
        self.pop_last = pts[-1][1] if pts else None


def cand_persistence(ctx):
    return list(ctx.last_logit)


def cand_curve(ctx):
    """Replica of the house fitted-curve baseline: per trophy, least squares
    of logit rate on ln(1 + hours since launch) over the last 24 h of reads,
    read at the target; fewer than 3 reads falls back to persistence."""
    out = list(ctx.last_logit)
    if not ctx.reads:
        return out
    t_last_read = ctx.reads[-1][0]
    win = [(t, rates) for t, rates in ctx.reads if t >= t_last_read - 24.0]
    if len(win) < 3:
        return out
    x_t = math.log1p(max(ctx.t_target, -0.9))
    for i in range(ctx.ntr):
        pts = [(math.log1p(max(t, -0.9)), logit_pct(rates[i]))
               for t, rates in win if i < len(rates) and rates[i] is not None]
        if len(pts) < 3 or ctx.last_logit[i] is None:
            continue
        a, b, _ = ols([p[0] for p in pts], [p[1] for p in pts])
        out[i] = clamp(a + b * x_t, -13.0, 13.0)
    return out


def cand_trend(ctx):
    """Damped log-time trend: OLS of logit rate on ln(1 + hours) over the
    recent distinct states, anchored at the last observed logit, with the
    extrapolated move shrunk by span / (span + horizon)."""
    out = list(ctx.last_logit)
    states = ctx.states
    if len(states) < 3:
        return out
    span = ctx.t_last - states[0].t
    win = [st for st in states if st.t >= ctx.t_last - min(72.0, span)]
    if len(win) < 3:
        win = states
    damp = (ctx.t_last - win[0].t) / max(1e-9, (ctx.t_last - win[0].t) + ctx.horizon_h)
    x_last = math.log1p(max(ctx.t_last, -0.9))
    x_t = math.log1p(max(ctx.t_target, -0.9))
    for i in range(ctx.ntr):
        if ctx.last_logit[i] is None:
            continue
        pts = [(math.log1p(max(st.t, -0.9)), logit_pct(rate_at(st, i)))
               for st in win if rate_at(st, i) is not None]
        if len(pts) < 3:
            continue
        _, b, _ = ols([p[0] for p in pts], [p[1] for p in pts])
        delta = clamp(b * (x_t - x_last) * damp, -2.5, 2.5)
        out[i] = ctx.last_logit[i] + delta
    return out


def cand_decomp(ctx):
    """Decomposition in log-time.  With x = ln(1 + hours since launch),
    d logit(rate)/dx = (a_x - b_x) / (1 - rate), where a_x is the trophy's
    earner growth per unit of log-time and b_x the tracked population's.
    Log-time is the natural clock here: growth that looks explosive in clock
    time right after launch decelerates as log-time advances, so integrating
    in x is self-damping where integrating in t is not.  a_x is estimated
    over the recent window (the launch transient excluded), shrunk toward
    the cross-trophy median by earner count, and decayed in x as it is
    integrated; b_x comes from the fitted population model; a soft ceiling
    a few times the current rate brakes saturation."""
    states = ctx.states
    if len(states) < 2 or ctx.pop is None:
        return None
    out = list(ctx.last_logit)

    def x_of(t):
        return math.log1p(max(t, -0.9))

    span = ctx.t_last - states[0].t
    w_hours = min(48.0, span)
    w_start = ctx.t_last - w_hours
    if span > 30.0:
        # The first hours are the launch transient (earners exploding from a
        # zero base); a window that includes them overstates growth.  The
        # final fit's 48 h window starts after it; prefixes should match.
        w_start = max(w_start, states[0].t + min(10.0, span * 0.25))
    win = [st for st in states if st.t >= w_start]
    if len(win) < 2:
        win = states[-2:]
    s0, s1 = win[0], win[-1]
    dx_win = max(1e-6, x_of(s1.t) - x_of(s0.t))
    b_recent_x = 0.0
    if s0.n and s1.n:
        b_recent_x = max(0.0, (math.log(s1.n) - math.log(s0.n)) / dx_win)
    a_raw = []
    for i in range(ctx.ntr):
        e0, e1 = earners_at(s0, i), earners_at(s1, i)
        if e1 is None:
            a_raw.append(None)
        else:
            a_raw.append((math.log(e1 + 0.5) - math.log((e0 or 0.0) + 0.5)) / dx_win)
    a_med = median_of([a for a in a_raw if a is not None])
    if a_med is None:
        a_med = 0.0
    x_last = x_of(ctx.t_last)
    x_target = x_of(ctx.t_target)
    steps = 24
    dx = (x_target - x_last) / steps if x_target > x_last else 0.0
    for i in range(ctx.ntr):
        lg0 = ctx.last_logit[i]
        if lg0 is None:
            continue
        a_i = a_raw[i] if a_raw[i] is not None else a_med
        e_last = ctx.last_earners[i]
        w = (e_last / (e_last + 80.0)) if e_last is not None else 0.3
        a_i = clamp(w * a_i + (1.0 - w) * a_med, -0.5, 3.0)
        r_last = ctx.last_rate[i]
        ceiling = min(60.0, max(r_last * 1.8, r_last + 4.0))
        cur = lg0
        for k in range(steps):
            x_mid = x_last + (k + 0.5) * dx
            t_mid = math.expm1(x_mid)
            b_x = clamp(pop_log_deriv(ctx.pop, t_mid) * (1.0 + max(t_mid, -0.9)),
                        0.0, max(b_recent_x * 1.25, 1e-4))
            a = a_i * math.exp(-(x_mid - x_last) / 1.2)
            r_pct = inv_logit_pct(cur)
            if a > 0.0 and r_pct > ceiling:
                a *= max(0.0, 1.0 - r_pct / ceiling)
            cur += (a - b_x) / max(1e-6, 1.0 - r_pct / 100.0) * dx
        out[i] = clamp(cur, lg0 - 2.0, lg0 + 1.5)
    return out


CANDIDATES = (cand_persistence, cand_curve, cand_trend, cand_decomp)


# ---------------------------------------------------------------------------
# Rolling-origin backtest inside the history: candidate weights + spread
# ---------------------------------------------------------------------------


def pinball(median_logit, sigma, y_logit, levels):
    total = 0.0
    for tau in levels:
        q = median_logit + _NORM.inv_cdf(tau) * sigma
        d = y_logit - q
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(levels)


def backtest(case, epoch, states, reads, levels):
    """Fit each candidate on prefixes of the history and score it against the
    final distinct state.  Returns (weights, ensemble residuals in logits)."""
    n = len(states)
    y = [logit_pct(rate_at(states[-1], i)) if rate_at(states[-1], i) is not None else None
         for i in range(len(case["trophies"]))]
    lo = max(3, n // 3)
    origins = [j for j in range(lo, n - 1)]
    if len(origins) > 12:
        step = len(origins) / 12.0
        origins = sorted({origins[int(k * step)] for k in range(12)})
    if not origins:
        return None, [], 0.0
    t_final = states[-1].t
    losses = [0.0] * len(CANDIDATES)
    counts = [0] * len(CANDIDATES)
    cache = []  # per origin: (candidate medians, horizon_days)
    horizon_days = []
    for j in origins:
        prefix = states[:j + 1]
        pre_reads = reads[:prefix[-1].end_idx + 1]
        ctx = Ctx(case, epoch, prefix, pre_reads, t_final)
        sigma = 0.5 * math.sqrt(max(0.0, (t_final - prefix[-1].t)) / 24.0)
        meds = []
        # Recency weight: an origin late in the history is fit on data whose
        # dynamics resemble the final fit's; early origins are dominated by
        # the launch transient and would misjudge the candidates.
        w_origin = (j / max(1, n - 1)) ** 2
        for k, cand in enumerate(CANDIDATES):
            m = cand(ctx)
            meds.append(m)
            if m is None:
                continue
            tot, cnt = 0.0, 0
            for i in range(ctx.ntr):
                if m[i] is None or y[i] is None:
                    continue
                tot += pinball(m[i], max(sigma, 0.05), y[i], levels)
                cnt += 1
            if cnt:
                losses[k] += w_origin * tot / cnt
                counts[k] += w_origin
        cache.append((meds, max(0.0, (t_final - prefix[-1].t) / 24.0)))
        horizon_days.append(max(0.0, (t_final - prefix[-1].t) / 24.0))
    if not any(counts):
        return None, [], 0.0
    avg = [losses[k] / counts[k] if counts[k] else 1e9 for k in range(len(CANDIDATES))]
    best = min(avg)
    weights = []
    for k in range(len(CANDIDATES)):
        if avg[k] > best + 0.18:
            weights.append(0.0)
        else:
            weights.append(math.exp(-(avg[k] - best) / 0.025))
    tot_w = sum(weights)
    if tot_w <= 0:
        return None, [], 0.0
    weights = [w / tot_w for w in weights]
    residuals = []
    for meds, _ in cache:
        for i in range(len(y)):
            if y[i] is None:
                continue
            num, den = 0.0, 0.0
            for k in range(len(CANDIDATES)):
                if weights[k] > 0 and meds[k] is not None and meds[k][i] is not None:
                    num += weights[k] * meds[k][i]
                    den += weights[k]
            if den > 0:
                residuals.append(y[i] - num / den)
    mean_horizon = sum(horizon_days) / len(horizon_days) if horizon_days else 0.0
    return weights, residuals, mean_horizon


# ---------------------------------------------------------------------------
# Frozen history (a single distinct state): the dilution-aware prior
# ---------------------------------------------------------------------------


def frozen_forecast(case, states, t_target, levels):
    """No trend exists to fit: the page has shown one state the whole window.

    The answering read shows the page as of Exophase's last refresh before the
    target, so the forecast is a two-branch mixture.  Branch A (no refresh
    lands first) is the frozen state itself, near-exact.  Branch B (a refresh
    lands) is built in earners space from the dynamics the kit's own
    competition notes publish for the first refresh: the tracked population
    multiplied (638 -> 3,868, x6.06) while total earners grew far less -- the
    tutorial trophy accounts for ~3,868 of the 4,276 earners at the refresh
    (every tracked player holds it), so the other 54 trophies' earners grew
    only 305 -> ~408 in total (+34%) while their denominator sextupled, and
    every non-tutorial rate fell, to roughly 0.15-0.3x its frozen value.

    Branch B therefore: tracked jumps by a factor G (several-fold for a young
    freeze, shrinking as the launch surge ages); the tutorial trophy stays at
    ~100% (its earners are the whole tracked population); the other trophies'
    earners grow modestly in total, the growth allocated across trophies in
    proportion to current earners with a floor, tilted toward the early-game
    (high-rate) trophies that progressing players reach first; each rate is
    then (earners grown) / (tracked x G).  Branch B's spread covers the
    uncertainty in G and in the allocation, wider for trophies with few
    earners.  Quantiles of the mixture are inverted exactly by bisection."""
    st = states[-1]
    ntr = len(case["trophies"])
    rates = [rate_at(st, i) for i in range(ntr)]
    horizon_h = max(0.0, t_target - st.t)
    horizon_d = horizon_h / 24.0
    # How long the state has already been observed unchanged.
    hist = case.get("history") or []
    if hist:
        epoch = launch_epoch(case)
        first_t = (parse_time(hist[0]["at"]) - epoch) / 3600.0
        at_t = (parse_time(case["at"]) - epoch) / 3600.0 if case.get("at") else st.t
        window_h = max(0.0, at_t - first_t)
    else:
        window_h = 0.0
    # Refresh hazard: refreshes land hours apart (~4-5 h); a state already
    # frozen for much of that interval is due one.
    remaining = max(0.6, 4.2 - 0.9 * window_h)
    p_refresh = clamp(1.0 - math.exp(-horizon_h / remaining), 0.0, 0.97)
    age_h = max(0.0, st.t)
    # Population jump at a refresh, shrinking as the launch surge ages.
    growth = 1.0 + 5.2 * math.exp(-age_h / 30.0)
    # Total earners growth (non-tutorial) by the refresh: +34% at the first
    # refresh, less (proportionally) as the base grows with age.
    earn_growth = clamp(0.34 * math.exp(-age_h / 48.0) + 0.04, 0.05, 0.40)
    # Earners per trophy now, and the allocation of the growth across them.
    earners = [earners_at(st, i) for i in range(ntr)]
    observed = sorted(r for r in rates if r is not None)
    pcts = []
    for r in rates:
        if r is None or not observed:
            pcts.append(0.5)
        else:
            below = sum(1 for x in observed if x < r)
            equal = sum(1 for x in observed if x == r)
            pcts.append((below + 0.5 * equal) / len(observed))
    plain = [i for i in range(ntr)
             if rates[i] is not None and rates[i] < 99.9 and earners[i] is not None]
    total_e = sum(earners[i] for i in plain)
    alloc_w = {i: (earners[i] + 2.0) * (0.6 + 0.8 * pcts[i]) for i in plain}
    total_w = sum(alloc_w.values())
    grown = {}
    for i in plain:
        share = alloc_w[i] / total_w if total_w > 0 else 0.0
        grown[i] = earners[i] + earn_growth * total_e * share
    s_a = 0.10 + 0.03 * math.sqrt(horizon_d)
    rows = []
    for i in range(ntr):
        r = rates[i]
        if r is None:
            med = logit_pct(0.3)
            qs = [med + _NORM.inv_cdf(tau) * 1.6 for tau in levels]
        elif r >= 99.9:
            # The tutorial trophy: held by essentially every tracked player,
            # before the refresh and after it alike (99.99% a day later).
            med = logit_pct(r)
            qs = [med + _NORM.inv_cdf(tau) * 0.2 for tau in levels]
        else:
            if i in grown and st.n:
                r_b = 100.0 * grown[i] / (st.n * growth)
            elif r > 0.0:
                # No population count to work in earners space: fall back to
                # a retention factor in rate space.
                kappa = 0.22 + 0.58 * pcts[i]
                r_b = r * (1.0 + kappa * (growth - 1.0)) / growth
            else:
                r_b = 0.05
            mu_a = logit_pct(r) if r > 0.0 else logit_pct(0.02)
            mu_b = logit_pct(r_b)
            e_i = earners[i] if earners[i] is not None else 0.0
            s_b = clamp(0.45 + 0.9 / math.sqrt(e_i + 4.0), 0.5, 1.1) \
                + 0.18 * math.sqrt(max(0.0, horizon_d - 0.1))
            qs = []
            for tau in levels:
                lo, hi = -14.0, 14.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    cdf = (1.0 - p_refresh) * phi((mid - mu_a) / s_a) \
                        + p_refresh * phi((mid - mu_b) / s_b)
                    if cdf < tau:
                        lo = mid
                    else:
                        hi = mid
                qs.append(0.5 * (lo + hi))
        rows.append(qs)
    return rows


# ---------------------------------------------------------------------------
# Multi-state forecast: weighted ensemble + calibrated spread
# ---------------------------------------------------------------------------


def ensemble_forecast(case, epoch, states, reads, t_target, levels):
    ntr = len(case["trophies"])
    ctx = Ctx(case, epoch, states, reads, t_target)
    meds = [cand(ctx) for cand in CANDIDATES]
    weights, residuals, bt_horizon_d = (None, [], 0.0)
    if len(states) >= 6:
        weights, residuals, bt_horizon_d = backtest(case, epoch, states, reads, levels)
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
        if meds[3] is None:
            weights = [0.34, 0.33, 0.33, 0.0]
    # Fallback medians for trophies no candidate can speak for.
    fallback = median_of([r for r in ctx.last_rate if r is not None])
    fallback_logit = logit_pct(fallback) if fallback is not None else logit_pct(0.5)
    horizon_d = ctx.horizon_h / 24.0
    # Spread: the ensemble's backtest residuals (RMS logit error at the
    # backtest horizons) grown to this case's horizon with a sub-square-root
    # exponent (much of the residual is cross-trophy dispersion, which does
    # not grow like a random walk), blended with the random-walk spread;
    # floored and capped.
    sigma_rw = 0.5 * math.sqrt(max(horizon_d, 0.0))
    if residuals:
        rms = math.sqrt(sum(x * x for x in residuals) / len(residuals))
        scale = clamp((max(horizon_d, 0.01) / max(bt_horizon_d, 0.25)) ** 0.35, 0.9, 1.8)
        sigma = clamp(0.65 * rms * scale + 0.35 * sigma_rw * 1.05, 0.35, 2.4)
    else:
        sigma = clamp(sigma_rw * 1.15, 0.35, 2.4)
    rows = []
    for i in range(ntr):
        num, den = 0.0, 0.0
        for k in range(len(CANDIDATES)):
            if weights[k] > 0 and meds[k] is not None and meds[k][i] is not None:
                num += weights[k] * meds[k][i]
                den += weights[k]
        med = num / den if den > 0 else fallback_logit
        s = sigma
        if ctx.last_rate[i] is not None and ctx.last_rate[i] >= 99.9:
            s = min(s, 0.35)
        elif ctx.last_earners[i] is not None and ctx.last_earners[i] < 5.0:
            s *= 1.25
        rows.append([med + _NORM.inv_cdf(tau) * s for tau in levels])
    return rows


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    epoch = launch_epoch(case)
    states = build_states(case, epoch)
    t_target = (parse_time(case["target"]) - epoch) / 3600.0 if case.get("target") \
        else (parse_time(case["at"]) - epoch) / 3600.0 + case.get("horizon_minutes", 0) / 60.0
    if not states:
        # No history at all: a wide generic prior for every trophy.
        rows = [[logit_pct(0.5) + _NORM.inv_cdf(tau) * 1.8 for tau in levels]
                for _ in case["trophies"]]
    elif len(states) == 1:
        rows = frozen_forecast(case, states, t_target, levels)
    else:
        reads = [((parse_time(rd["at"]) - epoch) / 3600.0, list(rd.get("rates") or []))
                 for rd in case.get("history") or []]
        rows = ensemble_forecast(case, epoch, states, reads, t_target, levels)
    out = []
    for trophy, qs in zip(case["trophies"], rows):
        vals = []
        prev = 0.0
        for lg in qs:
            v = clamp(inv_logit_pct(lg), 0.0, 100.0)
            v = max(v, prev)
            vals.append(round4(v))
            prev = float(vals[-1])
        out.append({"id": trophy["id"], "q": vals})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
