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

### Architecture competition: three philosophies competing, not one hard-coded

Rather than committing to either "shared multiclass models" or "10 independent
digit specialists" upfront, Astra runs **three** competing architectures per
symbol, side by side, and empirically promotes whichever wins
(`learning/architecture_competition.py`):

- **A. Global** -- the shared multiclass models + adaptive per-digit
  weighting described above (`decision/decision_engine.py::SymbolPipeline`).
- **B. Specialist** -- 10 independent per-digit binary specialists
  (`models/digit_specialist.py`), each a small Beta-Bernoulli +
  online-logistic ensemble estimating P(digit=d) as its own binary problem,
  combined into a distribution by renormalization.
- **C. Hybrid** -- an adaptively-weighted blend of A's and B's output
  (reuses the same per-digit `PerformanceTracker` machinery, just with
  "global" and "specialist" as its two "models").

Only the current **champion** architecture drives real trades. The other two
run in **shadow mode** every tick: their probability vectors are scored
against the exact same live quotes and the exact same trade-quality gates a
real decision would use
(`decision/decision_engine.py::evaluate_architecture_decision` -- the core
decision logic was extracted into a pure function specifically so all three
architectures get identical treatment, never special-cased). This mirrors
the shadow-trading approach already used elsewhere in this account and never
risks money on the non-champion architectures.

All seven requested metrics are tracked per architecture
(`ArchitectureMetrics`): log loss, Brier score, calibration (a reused
`CalibrationTracker` per architecture per side), probability stability
(tick-to-tick vector movement), internal model agreement (Global: across its
8 models; Specialist: between its Bayesian and logistic signals; Hybrid:
between Global's and Specialist's output), economic EV (unfiltered -- "how
good is the stated edge on average"), and realized/hypothetical trading
performance (P&L from only the trades that would have passed the gates).
Every `evaluate_every_n_ticks` (default 200), the three are compared on a
config-weighted composite score (each metric min-max normalized across the
three, direction-corrected) and the champion is only replaced if a
challenger wins with a real margin (`min_promotion_margin`) **and** the win
holds up across both halves of the evaluation window -- the same
adversarial stability discipline as `learning/champion_challenger.py`,
applied one level up.

Verified in `tests/test_architecture_competition.py`: promotion actually
fires when an architecture is genuinely and consistently better, and is
correctly refused when an apparent improvement doesn't hold up across both
halves of the window (a classic "looks good on average, but got there by
being great in the first half and bad in the second" case).

Note the within-Global weight-vector champion/challenger
(`learning/champion_challenger.py`) keeps running independently of this --
the two are complementary: architecture competition decides WHICH
architecture drives trades, while champion/challenger keeps tuning Global's
own weights regardless of whether Global is currently in the lead.

### Concurrent trade cap

`risk/risk_engine.py` now enforces a hard cap on simultaneously OPEN
(unsettled) contracts, **across every symbol**, not a per-symbol limit --
default 2 (`MAX_CONCURRENT_TRADES` env var / `risk.max_concurrent_trades` in
config.yaml). `app/main.py` claims a slot (`reserve_trade_slot()`)
synchronously, with no `await` between the risk check and the reservation,
so no other symbol's worker task can slip a trade through the gap under
asyncio's cooperative scheduling -- and always releases it in a
`finally` block so a slot can never leak on an execution error.

### Symbols: R_100 and 1HZ100V, with room to grow

`ASTRA_SYMBOLS` now defaults to `1HZ10V,1HZ100V` instead of being blank.
Adding more symbols later is a one-line env var change (comma-separated),
no code change needed; clearing it entirely reverts to full dynamic
discovery of every `R_*`/`1HZ*` synthetic index Deriv offers, which is still
fully implemented and available.

### Deployment fixes from a real Railway run

A live deployment surfaced two real bugs, both fixed and covered by tests:

1. **The Deriv Options API auth/request flow changed from what earlier
   testing assumed.** Fixed in `ingestion/deriv_client.py`:
   - Auth is now a two-step REST flow: `GET /trading/v1/options/accounts`
     (with `Deriv-App-ID` + `Authorization: Bearer <token>` headers) to
     resolve an account, then `POST /trading/v1/options/accounts/{id}/otp`
     (no JSON body) for a pre-authenticated WS URL. The account resolver
     picks a demo or real account automatically (`DERIV_USE_REAL`, default
     demo) or uses a pinned `DERIV_ACCOUNT_ID`.
   - `active_symbols` requests must NOT include `"product_type"` anymore
     (current API rejects it), and the response field is `underlying_symbol`,
     not `symbol`.
   - `proposal` requests must send `underlying_symbol`, not `symbol`
     (`InputValidationFailed: Properties not allowed: symbol` otherwise --
     this was the cause of 531 consecutive "Proposal request failed" log
     lines in the deployment log that surfaced it; every single trade
     evaluation was failing to price a contract).
   - New `DRY_RUN` x `DERIV_USE_REAL` mode matrix (see `.env.example`):
     demo+dry-run for a first connectivity smoke test, demo+live for
     "paper trading that exercises the real order flow" (recommended before
     risking money), real+live for actual trading, real+dry-run for a
     real-account balance/access check with no orders.
2. **numpy booleans silently broke Supabase logging.** Comparing two
   `numpy.float64` values (as `learning/champion_challenger.py` and
   `learning/architecture_competition.py` both do when checking rolling
   log-loss improvement) produces `numpy.bool` -- which, unlike
   `numpy.float64`, does NOT subclass Python's built-in `bool` (`bool` can't
   be subclassed at all), so it fails `json.dumps` with "Object of type bool
   is not JSON serializable". Because `database/repository.py`'s `_safe()`
   wrapper catches and logs write failures rather than crashing, this
   silently zeroed out `astra_experiment_log` for the entire deployment with
   no other symptom. Fixed at the source (explicit `float()`/`bool()` casts
   in `champion_challenger.py`) AND defensively (`database/repository.py`
   now runs every payload through `_json_safe()`, which recursively converts
   any numpy scalar/array to a native Python type before it reaches
   Supabase) so this class of bug can't silently recur from anywhere else,
   including the new architecture-competition metrics. Covered by
   `tests/test_json_safe.py`, which reproduces the exact failure mode.

### Tick summary logging (debug at a glance)

Every 150 ticks, each symbol worker logs (and persists to
`astra_system_events`) a rolling summary (`app/tick_summary.py`):
trades executed, wins/losses/P&L, and -- if trades were scarce or zero --
the top specific reasons why, broken down per gate (e.g.
`insufficient_edge: 40, quality_score_below_minimum: 30`) rather than one
opaque combined string. Trades blocked by the risk engine (e.g. the
concurrent-trade cap) show up as `risk_blocked:<reason>` in the same
breakdown, distinct from decision-engine gating. A Railway log skim now
answers "is Astra actually trading, and if not, which specific gate is the
bottleneck" without needing to query Supabase or read raw `reason` strings
by hand. Covered by `tests/test_tick_summary.py`.
