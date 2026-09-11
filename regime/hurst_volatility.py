"""
Hurst exponent and realized-vol-vs-baseline: the two inputs
decision/regime_conviction.py's classify_regime() needs that Astra doesn't
compute anywhere yet.

hurst_rs() is ported faithfully, including a real bugfix already baked into
the source (see its docstring): computed on raw PRICES instead of returns,
Hurst on a random walk spuriously converges to ~1.0 regardless of true
dynamics (prices never mean-revert, so R/S analysis on them always reads
"maximally trending") -- this produced a permanent, structural directional
bias in the source's fusion layer. Computing on log-returns (stationary,
zero-mean) instead is the fix, and is exactly why this takes PriceSeries's
already-computed tick_log_returns/minute_log_returns rather than raw prices.

realized_vol_and_baseline() ports the sigma_now/sigma_baseline computation
pattern used at the source's regime_decision() call site (not a named
function there -- inlined at the call site, extracted here into its own
function for testability): sigma_now is the std of the most recent window;
sigma_baseline is a MAD-based (median absolute deviation) estimate over a
longer window, scaled by 1.2533 (the standard MAD-to-std conversion factor
for a Gaussian: E[|X|] = sigma*sqrt(2/pi), so sigma = E[|X|]/sqrt(2/pi) =
E[|X|]*1.2533) -- more outlier-robust than a plain std over the same window,
which is the point of using it specifically as the BASELINE.
"""
from __future__ import annotations

import numpy as np


def hurst_rs(log_returns, min_window: int = 10) -> float:
    """Rescaled-range Hurst exponent on a log-return series. Returns 0.5
    (the "no information" value -- neither trending nor mean-reverting)
    when there isn't enough history for a meaningful estimate, rather than
    raising -- callers should treat 0.5 as "regime unknown", which
    classify_regime() already does correctly (0.5 falls in the neutral
    band by default).
    """
    series = np.asarray(log_returns, dtype=float)
    n = len(series)
    if n < 50:
        return 0.5
    max_window = n // 2
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int)
    )
    rs_points = []
    for w in window_sizes:
        n_chunks = n // w
        if n_chunks < 1:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = series[i * w:(i + 1) * w]
            mean = np.mean(chunk)
            dev = np.cumsum(chunk - mean)
            r = np.max(dev) - np.min(dev)
            s = np.std(chunk)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_points.append((w, np.mean(rs_chunk)))
    if len(rs_points) < 3:
        return 0.5
    log_w = np.log([w for w, _ in rs_points])
    log_rs = np.log([rs for _, rs in rs_points])
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def realized_vol_and_baseline(returns, now_window: int = 30,
                               baseline_window: int = 200) -> tuple[float, float]:
    """Returns (sigma_now, sigma_baseline) for classify_regime(). Both 0.0
    (classify_regime treats sigma_baseline<=0 as "no baseline yet" ->
    NEUTRAL, correctly refusing to classify a regime it can't support) when
    there isn't enough history yet.
    """
    returns = np.asarray(returns, dtype=float)
    sigma_now = float(np.std(returns[-now_window:])) if len(returns) >= now_window else 0.0
    if len(returns) >= max(60, now_window * 2):
        recent = returns[-baseline_window:]
        sigma_baseline = float(np.median(np.abs(recent))) * 1.2533
    else:
        sigma_baseline = 0.0
    return sigma_now, sigma_baseline
