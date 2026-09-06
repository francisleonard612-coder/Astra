# Astra -- Deriv Digit Probability / Mispricing Trading System

A live trading bot for Deriv's synthetic index digit contracts (DIGITOVER /
DIGITUNDER), built from the Astra master spec with three deliberate scope
decisions made for this build:

1. **Monitoring dashboard/alerting is out of scope.** Structured JSON logs
   (readable in Railway's log viewer) plus the Supabase tables are the
   source of truth. See `app/logging_setup.py`.
2. **Database is Supabase**, with the full SQL schema in `database/schema.sql`.
3. **Trades every discovered `R_*` and `1HZ*` synthetic index**, 1 tick
   duration, decided dynamically at startup (and refreshed periodically) via
   Deriv's `active_symbols` call -- not a hardcoded list. Override with the
   `ASTRA_SYMBOLS` env var if you want to test against a subset.

## What this is (and isn't)

This is a genuinely working implementation of the pipeline the spec
describes -- ingestion, feature engineering, a multi-model ensemble,
calibration, regime detection, live mispricing detection against real
payouts, risk management, execution, and online learning -- not a stub. It
has been tested (unit tests + a synthetic end-to-end backtest) and correctly
does two important things: it **abstains** from trading on pure noise, and
it **does** trade (and win, in the synthetic test) when there's a real,
well-calibrated edge.

Two places where this build pragmatically diverges from the spec's fullest
vision, documented in the code:

- **Champion/challenger** (`learning/champion_challenger.py`) operates on
  the per-symbol *ensemble weight vector*, not fully separate duplicated
  model architectures per digit specialist. This preserves the "production
  only uses what's earned promotion" property without a second full
  training pipeline.
- **The "research agent" layer** (director / model scientist / adversarial
  analyst, spec sections 24-27) is implemented as deterministic, rule-based
  evaluation (`research/experiment_log.py`, the promotion logic in
  `champion_challenger.py`) rather than literal AI agents -- consistent with
  the spec's own rule that no LLM calls belong in the trading path.
- **Per-digit specialists** (spec sections 5 & 7) are implemented as
  adaptive per-digit *weighting* of the shared models, not 10 separately
  instantiated model objects per digit. See the "Per-digit specialist
  weighting" section below for why and how it's verified.

## Setup

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DERIV_API_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY
```

Apply the database schema once, in the Supabase SQL editor (or via `psql`):

```bash
psql "$SUPABASE_DB_URL" -f database/schema.sql
# or paste database/schema.sql into the Supabase dashboard's SQL editor
```

Run it:

```bash
python -m app.main
```

Start in `DRY_RUN=true` first -- Astra will discover symbols, build up
state, evaluate real decisions, and log everything to Supabase, but will
never call Deriv's `buy`. Flip to `DRY_RUN=false` once you're happy with
what you see in `astra_predictions`.

### Deploy to Railway

This repo includes a `Procfile` and `railway.json` (worker process, no HTTP
port -- matches how the account's other Deriv bots are deployed). Push the
repo, set the env vars from `.env.example` in the Railway dashboard, and
deploy.

## Key config knobs (`configs/config.yaml`)

Everything research-y (feature windows, ensemble weights, calibration
method, regime thresholds, mispricing gates, quality score weights,
champion/challenger cadence) lives here so it's all in one readable place.
Credentials, stake sizing, and risk limits are environment variables (see
`.env.example`) so they can differ per-deployment without touching code.

Notably: **martingale staking defaults to OFF** (`risk.staking.enabled:
false`). A sibling bot in this account (`digit_over_bot`) ran martingale
live for 278 trades and lost more from it than flat staking would have --
the top stake tier had the same win rate as the base tier, so it just
amplified losses when the underlying edge didn't hold. Astra's staking
module (`risk/staking.py`) is still there and config-driven if you want to
re-test it, but turning it on isn't the default for a reason.

## Architecture

```
ingestion/    Deriv WS client (OTP token-exchange auth, tick queue+worker --
              see the deriv_client.py docstring for the deadlock bug this
              avoids, found in a sibling bot)
state/        Per-symbol rolling digit history
features/     Frequency, gap, streak, transition, entropy features
models/       uniform, rolling/EWMA frequency, Bayesian, Markov (order
              backoff), online logistic (SGD), Random Forest, XGBoost,
              ensemble fusion, isotonic/Platt calibration
regime/       Data-driven regime classification
pricing/      Live payout -> breakeven -> edge -> mispricing gate
decision/     Per-tick decision engine + trade quality scoring
risk/         Hard risk limits (independent of model opinion) + staking
execution/    Re-validates quotes, buys, waits for settlement
learning/     Online performance-weighted ensemble, batch model retraining
              cadence, champion/challenger promotion
research/     Deterministic experiment log for challenger evaluations
database/     Supabase schema + repository (every write is best-effort --
              a DB hiccup never crashes the trading loop)
backtest/     Causal tick-replay simulator (see its IMPORTANT LIMITATION
              note below before trusting any backtest P&L number)
app/          Config, logging, main entrypoint
```

## Testing

```bash
pytest tests/ -v
```

32 unit tests cover feature engineering, Markov order backoff, ensemble
combination/agreement, pricing math, calibration (including the reliability
vs. skill distinction below), trade quality scoring, and risk engine limits.

### A real bug this testing process caught (worth knowing about)

The first version of the calibration quality gate scored a model's
reliability using a Brier-skill-score against the *observed base rate*. On
a synthetic stream with a real, strong digit bias, that gate stayed near
zero and silently blocked every trade -- because a model that (correctly)
predicts "the base rate" has, by definition, no *skill* beyond the base
rate, even though its stated probability is exactly right and very
tradeable against an exchange price that disagrees with it. Skill and
calibration reliability are different things (this is the classical Brier
score reliability/resolution decomposition), and the gate needs reliability,
not skill. It's now a binned Expected-Calibration-Error style reliability
score instead. This is exactly the kind of thing an end-to-end synthetic
backtest is for -- catching a plausible-looking gate that would have quietly
zeroed out every trade in production.

### Two performance bugs caught while adding per-digit specialist weighting

Adding per-digit weighting (below) motivated a closer look at per-tick cost,
which surfaced two real bugs that would have made Astra fall further and
further behind live ticks the longer it ran -- both fixed, both verified to
produce byte-identical trading decisions before and after:

1. **Markov transition counts were rescanned from scratch every tick.**
   `MarkovModel` used to rebuild its whole order-1/2/3 transition table by
   scanning the entire retained digit history (up to `max_window`, e.g.
   2500 ticks) on every single prediction. Fixed by moving to incremental
   O(max_markov_order) counting maintained directly in
   `state/rolling_state.py::SymbolState.push()` -- counts are updated (and
   correctly decremented on window eviction) as each digit arrives, so
   `MarkovModel` now does an O(1) dict lookup instead of an O(window) scan.
2. **Calibration quality scoring called the sklearn isotonic calibrator
   one element at a time, in a Python loop, over the whole buffer (up to
   2000 samples), every tick, for both Over and Under.** Profiling a
   3000-tick backtest showed this alone eating ~70% of total runtime.
   Fixed by batching it into one vectorized `predict()` call over the whole
   buffer, plus a small dirty-flag cache so it isn't recomputed at all
   between `record()` calls.

Net effect: a 3000-tick backtest went from timing out (>90s) to ~21s
(~7ms/tick) -- and produced **exactly** the same trade count, win count, and
P&L before and after both fixes, confirming these were pure performance
fixes with zero change to prediction or trading behavior. At real trading
cadence (ticks arriving roughly once per second per symbol), 7-10ms of
compute per tick has enormous headroom.

### Per-digit specialist weighting

The spec (sections 5 & 7) asks for 10 digit specialists -- e.g. digit 0
might be best called by Bayesian+Markov+XGBoost while digit 4 is best
called by transition+GBM+frequency. This is now implemented: every model
still predicts the full 10-digit vector (shared feature representation, as
section 7 also calls for), but `learning/online.py::PerformanceTracker` now
tracks a separate rolling *binary* log-loss per `(model, digit)` pair and
produces a length-10 weight vector per model instead of one scalar weight.
`models/ensemble.py::combine()` accepts either (backward compatible).

**Scope note:** this does NOT instantiate 10 separate trained model objects
per digit per symbol -- that would be 8 models x 10 digits x N symbols of
independently-fitted objects, a real memory cost on a Railway worker
running every `R_*`/`1HZ*` symbol at once. Adaptive per-digit weighting of
shared models achieves the same practical outcome ("digit 4 ends up mostly
listening to Markov+frequency") at a fraction of the footprint. If you want
literal separate fitted specialist objects per digit instead, that's a
bigger follow-up.

Verified with `tests/test_online_learning.py`: weights sum to 1 per digit
across models, a model that's specifically good at one digit gets
upweighted there and NOT elsewhere, and `combine()` handles both the old
scalar and new per-digit weight shapes.

### Learning begins immediately

`min_samples_per_symbol` (default 300) gates *trading* decisions, not
*learning*. Verified in `tests/test_learning_starts_immediately.py`:
Markov transition counts accumulate from the very first tick, the online
logistic model (`SGDClassifier.partial_fit`) starts fitting from the second
tick (the first realized outcome), and per-digit performance tracking has
live data well before the 300-sample trading threshold. Only the batch
models (Random Forest / XGBoost) wait -- they genuinely need enough data to
avoid overfitting -- and even then, `learning/retraining.py` now fires their
*first* fit as soon as they have enough buffered samples rather than
additionally waiting for the next retrain-cadence boundary on top of that.

### The backtest simulator's one real limitation

`backtest/simulator.py`'s synthetic contract quote uses one flat payout
ratio for every barrier. Real Deriv payouts are priced per barrier (a
barrier of 2 has a much higher win probability, hence lower payout, than a
barrier of 8), so a backtest run can look "profitable" purely from a
barrier's own built-in win-rate geometry against an unrealistic flat
payout assumption -- not a real edge. Use the backtester to sanity-check
that the pipeline's plumbing behaves correctly (does it abstain on noise?
does it fire and win on an injected bias?), not to estimate real returns.
Live trading is unaffected by this -- `execution/orders.py` and
`pricing/payout.py` always fetch a real proposal from Deriv immediately
before every decision and again before every buy.

## Database volume note

With many symbols each ticking roughly once a second, `astra_ticks` and
`astra_predictions` can generate a lot of rows fast. Defaults: raw ticks are
persisted per config, but predictions are only fully logged when a trade
actually fires, plus a 1-in-20 sample of `NO_TRADE` ticks (see
`PREDICTION_LOG_SAMPLE_EVERY_N` in `app/main.py`) so you still get visibility
into why Astra is abstaining without logging every single tick. Adjust that
constant, and `database.tick_retention_hours` in `configs/config.yaml`
(ticks older than this get pruned), to taste.
