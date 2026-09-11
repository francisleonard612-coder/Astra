"""
Monte Carlo win-probability estimation for a given (direction, duration).

Ported from risefall-bot/risefall_bot_v4_hmm_gbm.py's monte_carlo_duration()
and its supporting hmm_gbm_terminal_log_returns()/_hmm_state_params(), with
the v10 duration-bias fixes kept intact -- see the comments inline, which
are preserved because they explain WHY the math is shaped this way, not just
what it does; losing them would make this code look like it could safely be
"simplified" back into the exact bug it fixes.

One additional fix made DURING this port, not present in the source: a
direction-double-application bug found by testing this against both
directions on the same returns (something the source never does -- see the
"PORT-TIME FIX" comment on the win-count line below for the full
explanation and how it was found/verified).

CRITICAL: never mix tick-duration and minute-duration candidates in one call.
`returns` here is a single fixed-resolution series (all ticks, or all
minute-bar closes) -- a "duration=10" candidate means 10 steps of WHATEVER
resolution `returns` is. Astra's own risefall-bot history has a whole
postmortem about exactly this (see the source repo's README, "v10: minutes
only"): sweeping both tick and minute candidates through one MC call, where
drift is a noisy point estimate that gets projected forward by `duration`
while diffusion noise only grows by sqrt(duration), mechanically biased
selection toward the longest/coarsest candidate on pure noise -- worse the
wider the duration range spans. Call this once per resolution (tick returns
+ tick candidates, then separately minute returns + minute candidates) and
compare the two results' blended win-probabilities to choose between them,
never pool the candidates into one call.

Not yet wired in from Astra's side: `hmm_model` (a per-symbol fit
hmmlearn.GaussianHMM on returns -- Astra's regime/detector.py is digit-
frequency-based, not a returns HMM) and `feats` entries for OU mean-reversion
pull / Hawkes jump-clustering signal, all upstream additions from later
phases. Every one of those degrades gracefully to a no-op when absent (that
was already true of the original code, not something added during the
port), so this module is fully correct today with hmm_model=None and
feats={} -- just without those refinements yet.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_MC_SIMULATIONS = 5000


def _hmm_state_params(hmm_model) -> tuple[np.ndarray, np.ndarray] | None:
    """Extracts (means, sds) per state from a fit hmmlearn GaussianHMM,
    ordered as hmmlearn returns them (component order is arbitrary but
    consistent within one fit -- no state labelling needed here, only
    per-state parameters and the filtered posterior)."""
    try:
        means = hmm_model.means_.flatten()
        sds = np.array([np.sqrt(c[0][0]) for c in hmm_model.covars_])
        if not (np.all(np.isfinite(means)) and np.all(np.isfinite(sds)) and np.all(sds > 0)):
            return None
        return means, sds
    except Exception:  # noqa: BLE001
        return None


def hmm_gbm_terminal_log_returns(hmm_model, recent_returns: np.ndarray, fallback_vol: float,
                                  n_steps: int, n_sims: int,
                                  rng: np.random.Generator | None = None,
                                  hmm_posterior_lookback: int = 200) -> np.ndarray:
    """Draws n_sims cumulative log-return samples over n_steps ticks/bars
    using an existing per-symbol hmmlearn model. Each simulated path draws
    ONE regime from the model's current filtered posterior and stays in it
    for the whole horizon -- regime persistence over a short horizon is a
    reasonable approximation given these models' typically high self-
    transition probabilities. Falls back to a flat single-Gaussian model if
    hmm_model is None or too few recent_returns are available for a
    posterior.
    """
    rng = rng or np.random.default_rng()
    params = _hmm_state_params(hmm_model) if hmm_model is not None else None
    if params is None or len(recent_returns) < 5:
        return rng.normal(0.0, max(fallback_vol, 1e-9) * math.sqrt(n_steps), size=n_sims)

    means, sds = params
    try:
        posterior = hmm_model.predict_proba(recent_returns.reshape(-1, 1))[-1]
    except Exception:  # noqa: BLE001
        return rng.normal(0.0, max(fallback_vol, 1e-9) * math.sqrt(n_steps), size=n_sims)

    if len(posterior) != len(means):
        return rng.normal(0.0, max(fallback_vol, 1e-9) * math.sqrt(n_steps), size=n_sims)

    state_idx = rng.choice(len(posterior), size=n_sims, p=posterior / posterior.sum())
    mu_path = means[state_idx] * n_steps
    # v10 FIX: same duration bias as monte_carlo_duration()'s Gaussian half
    # below -- `means` here are the HMM's FITTED per-state emission means,
    # themselves estimated from limited history via EM, not known-true
    # parameters. Projecting them forward by n_steps without accounting for
    # their own estimation uncertainty produces the same runaway-confidence-
    # at-long-duration artifact. hmm_posterior_lookback is used as a
    # conservative proxy for the effective sample size behind each state's
    # mean estimate (not exact -- the true per-state observation count isn't
    # tracked here -- but enough to stop the bias from compounding).
    effective_n = max(hmm_posterior_lookback, 10)
    mean_se = sds[state_idx] / math.sqrt(effective_n)
    drift_uncertainty = mean_se * n_steps
    diffusion_std = sds[state_idx] * math.sqrt(n_steps)
    sd_path = np.clip(np.sqrt(drift_uncertainty ** 2 + diffusion_std ** 2), 1e-9, None)
    return rng.normal(mu_path, sd_path)


def monte_carlo_duration(returns: np.ndarray, direction: int, candidate_durations: list[int],
                          n_sims: int = DEFAULT_MC_SIMULATIONS, hmm_model=None,
                          feats: dict | None = None, empirical_win_rates: dict | None = None,
                          prices: np.ndarray | None = None,
                          rng: np.random.Generator | None = None) -> tuple[int, float]:
    """Takes `direction` (+1 for Rise, -1 for Fall) as given -- this does NOT
    decide direction, only which candidate duration maximizes the estimated
    win probability for that direction. Astra's caller runs this twice, once
    per direction, and compares both against Deriv's quoted breakeven
    probability to find genuine mispricing (see
    decision/rise_fall_decision_engine.py's evaluate()) --
    a materially different use than the original bot's "pick the direction
    the Bayesian layer already chose", but the duration-selection math itself
    is unchanged.

    Returns (best_duration, best_win_probability_estimate).
    """
    feats = feats or {}
    empirical = empirical_win_rates or {}
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 20:
        return candidate_durations[0], 0.5

    rng = rng or np.random.default_rng()

    cond_vol = feats.get("cond_vol")
    vol = cond_vol if cond_vol and cond_vol > 0 else (
        float(np.std(returns[-50:])) if len(returns) >= 50 else float(np.std(returns)))
    vol = vol if vol > 0 else 1e-6

    hawkes_signal = feats.get("hawkes", 0.0)
    # v10 FIX (the primary one): this used to be
    # `direction * abs(np.mean(returns[-50:]))` -- taking the ABSOLUTE VALUE
    # of recent momentum and reapplying it in whatever direction was already
    # chosen. That forces E[drift] > 0 in the trade's favor even on PURE
    # NOISE (E[|X|] > 0 even when E[X]=0, for any non-degenerate X), and
    # since this drift is projected forward by `dur`, that constant one-
    # sided bias compounds with duration -- exactly the "MC favors long
    # durations" symptom. Using the SIGNED mean instead means a genuine
    # headwind shows up as a headwind, and E[drift] = 0 exactly on pure
    # noise, matching this function's own "vol/drift ~ 0 by design on these
    # instruments" assumption instead of silently violating it.
    drift = direction * float(np.mean(returns[-50:])) * (1 + abs(hawkes_signal) * 0.5) if len(returns) >= 50 else 0.0

    ou_params = feats.get("ou_params")
    trend_weight = feats.get("trend_weight", 0.5)
    reversion_pull = 0.0
    if ou_params and ou_params.get("theta", 0) > 0 and prices is not None and len(prices) > 0:
        current_price = prices[-1]
        raw_pull = ou_params["theta"] * (ou_params["mu"] - current_price) * 0.01
        # NOT direction-reoriented, unlike `drift` above -- reversion_pull is
        # a property of price vs. its estimated mean-reversion level,
        # independent of which direction is being evaluated. Currently
        # dormant (ou_params is never populated by Astra yet, so this whole
        # branch never runs), but flagging now: whoever wires OU reversion
        # in should double-check this against the same
        # both-directions-on-the-same-data test used to catch the
        # direction-double-application bug above (see the PORT-TIME FIX
        # comment on the win-count line) before trusting it, since mixing a
        # reoriented term (drift) with a non-reoriented one (reversion_pull)
        # in the same sum is exactly the kind of asymmetry that bug was.
        reversion_pull = raw_pull * (1 - trend_weight)

    hmm_posterior_lookback = feats.get("hmm_posterior_lookback", 200)
    recent_returns = returns[-hmm_posterior_lookback:] if len(returns) >= hmm_posterior_lookback else returns
    use_hmm = hmm_model is not None and _hmm_state_params(hmm_model) is not None

    # ---- Terminal displacement model -------------------------------------
    # Deriv Rise/Fall settles on price[expiry] vs price[entry]. The correct
    # model for the terminal displacement after `dur` independent steps is
    #   X_T ~ N(drift * dur, vol * sqrt(dur))
    # sampled DIRECTLY, not accumulated tick-by-tick (accumulating and then
    # testing sum(steps)>0 is mathematically equivalent for the terminal
    # value alone, but is where the original, pre-v10 bias entered if the
    # per-step draws weren't perfectly independent/identically distributed).
    #
    # v10 FIX (the second, deeper one): `drift` above is a POINT ESTIMATE
    # (mean of the last 50 returns), not a known-true parameter -- it
    # carries its own estimation uncertainty (standard error), and that
    # uncertainty must be propagated into the simulation. Since the drift
    # TERM scales as O(dur) while diffusion noise alone only scales as
    # O(sqrt(dur)), ANY nonzero drift estimate -- including one that's pure
    # sampling noise with no real predictive content -- mechanically
    # produces increasingly extreme (and increasingly WRONG) confidence as
    # `dur` grows, purely from projecting a noisy point estimate further
    # into the future. This is exactly why an MC built without this
    # correction ends up biased toward picking the longest candidate
    # duration regardless of whether longer durations are genuinely more
    # predictable.
    #
    # Fix: treat drift as an ESTIMATED parameter with its own standard error
    # (std of the estimation window / sqrt(window size)), and combine that
    # DRIFT uncertainty with diffusion uncertainty IN QUADRATURE for the
    # terminal std: sqrt((dur*drift_se)^2 + (vol*sqrt(dur))^2). The drift-
    # uncertainty term grows as O(dur) too now, so it catches up with and
    # eventually dominates the drift term itself at long durations,
    # correctly preventing runaway confidence in a noisy estimate instead of
    # rewarding it.
    drift_window = returns[-50:] if len(returns) >= 50 else returns
    n_drift = max(len(drift_window), 2)
    drift_se = float(np.std(drift_window)) / math.sqrt(n_drift)

    best = None
    for dur in candidate_durations:
        drift_uncertainty = drift_se * dur
        diffusion_std = vol * np.sqrt(dur)
        total_std = float(np.sqrt(drift_uncertainty ** 2 + diffusion_std ** 2))

        gaussian_terminal = rng.normal(
            (drift + reversion_pull) * dur,
            total_std,
            size=n_sims // 2 if use_hmm else n_sims,
        )
        if use_hmm:
            gbm_log_ret = hmm_gbm_terminal_log_returns(
                hmm_model, recent_returns, vol, dur, n_sims - n_sims // 2, rng=rng,
                hmm_posterior_lookback=hmm_posterior_lookback,
            )
            gbm_terminal = gbm_log_ret + (drift + reversion_pull) * dur
            terminal = np.concatenate([gaussian_terminal, gbm_terminal])
        else:
            terminal = gaussian_terminal

        # PORT-TIME FIX (found while testing this port, not present as a
        # known issue in the source material): the source computed
        # `wins = sum(terminal>0) if direction>0 else sum(terminal<0)` here.
        # `terminal`'s MEAN is already direction-oriented via `drift =
        # direction * raw_mean` above -- conditioning the win-check on
        # `direction` again applies the same sign flip a SECOND time. Net
        # effect, confirmed numerically: P(CALL wins) and P(PUT wins)
        # computed for the SAME returns collapse to the IDENTICAL value
        # regardless of genuine drift direction/magnitude -- the function
        # becomes direction-invariant, unable to distinguish a favorable
        # call from an unfavorable one. The source bot never surfaces this:
        # it only ever evaluates ONE pre-chosen direction per cycle (from an
        # independent Bayesian layer), so it never compares both directions
        # against the same returns where the collapse becomes visible.
        # Astra's use here does exactly that (decision/rise_fall_decision_
        # engine.py's evaluate() evaluates both RISE and FALL to find
        # mispricing), which is what surfaced it. Fix: since `terminal`'s
        # mean already encodes direction-adjusted favorability, the win check must be
        # unconditional (`terminal > 0`), not re-conditioned on direction.

        wins = np.sum(terminal > 0)
        sim_win_rate = wins / len(terminal)

        # Magnitude-weighted win rate: a naive win-count treats a path that
        # ends barely past zero the same as one that ends far in favour of
        # the direction. Borderline paths are weak evidence and inflate the
        # apparent edge; weighting by |terminal|/std down-weights borderline
        # crossings for a sharper, more honest conviction estimate.
        std_term = float(np.std(terminal)) + 1e-9
        weights = 1.0 + np.tanh(np.abs(terminal) / std_term)
        weighted_win_rate = float(np.sum(weights * (terminal > 0)) / np.sum(weights))

        sim_component = 0.5 * sim_win_rate + 0.5 * weighted_win_rate
        blended = (0.30 * sim_component + 0.70 * empirical[dur]
                   if dur in empirical and empirical[dur] > 0
                   else sim_component)
        if best is None or blended > best[1]:
            best = (dur, blended)
    return best
