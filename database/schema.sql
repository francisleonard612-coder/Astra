-- Astra schema for Supabase (Postgres).
-- Run this once in the Supabase SQL editor (or via `psql`) before starting
-- the bot. Safe to re-run: every statement is idempotent.
--
-- Scope note: the spec's full table list (ticks, predictions,
-- digit_predictions, contract_quotes, trades, trade_results, model_registry,
-- model_metrics, experiments, features, regime_states, calibration_records,
-- risk_events, system_events, drift_events) is consolidated here into a
-- smaller set of tables that cover the same information without
-- unnecessary duplication -- e.g. a single `astra_predictions` row already
-- carries the digit distribution, the derived Over/Under probabilities, the
-- regime, and the calibration/agreement scores that would otherwise live in
-- three separate tables.

create table if not exists astra_symbols (
    symbol text primary key,
    market text,
    first_seen_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    active boolean not null default true
);

-- Raw ticks, kept lean and pruned on a rolling basis (see
-- database/repository.py prune_old_ticks). Only stored when
-- database.persist_ticks is true in config.yaml.
create table if not exists astra_ticks (
    id bigserial primary key,
    symbol text not null references astra_symbols(symbol),
    epoch bigint not null,
    quote double precision not null,
    digit smallint not null check (digit between 0 and 9),
    created_at timestamptz not null default now()
);
create index if not exists idx_astra_ticks_symbol_created on astra_ticks (symbol, created_at desc);

-- One row per tick per symbol: the full decision object (spec section 44).
create table if not exists astra_predictions (
    id bigserial primary key,
    symbol text not null,
    ts timestamptz not null default now(),
    probabilities jsonb not null,          -- {"0": 0.09, "1": 0.08, ...}
    over_probability double precision,
    under_probability double precision,
    over_edge double precision,
    under_edge double precision,
    over_ev double precision,
    under_ev double precision,
    regime text,
    model_agreement jsonb,
    calibration_quality jsonb,
    quality_score double precision,
    decision text not null,
    reason text,
    sample_size integer,
    raw_model_predictions jsonb
);
create index if not exists idx_astra_predictions_symbol_ts on astra_predictions (symbol, ts desc);
create index if not exists idx_astra_predictions_decision on astra_predictions (decision);

create table if not exists astra_trades (
    id bigserial primary key,
    symbol text not null,
    ts timestamptz not null default now(),
    contract_type text not null,           -- DIGITOVER | DIGITUNDER | CALL | PUT | CALLE | PUTE
    barrier smallint,                      -- DIGITOVER/DIGITUNDER only; null for Rise/Fall (CALL/PUT)
    stake double precision not null,
    payout double precision,
    contract_id bigint,
    won boolean,
    pnl double precision,
    error text,
    prediction_id bigint references astra_predictions(id)
);
create index if not exists idx_astra_trades_symbol_ts on astra_trades (symbol, ts desc);

-- Run this against an existing database where astra_trades was already
-- created with the old `barrier smallint not null` definition -- without
-- it, every Rise/Fall trade insert (barrier=None) is rejected outright.
-- Safe to run even if the column is already nullable (no-op in that case).
alter table astra_trades alter column barrier drop not null;

create table if not exists astra_regime_log (
    id bigserial primary key,
    symbol text not null,
    ts timestamptz not null default now(),
    regime text not null,
    detail jsonb
);
create index if not exists idx_astra_regime_log_symbol_ts on astra_regime_log (symbol, ts desc);

create table if not exists astra_calibration_records (
    id bigserial primary key,
    symbol text not null,
    side text not null,                    -- OVER | UNDER
    ts timestamptz not null default now(),
    sample_size integer,
    quality_score double precision,
    rolling_log_loss double precision
);

create table if not exists astra_model_performance (
    id bigserial primary key,
    symbol text not null,
    ts timestamptz not null default now(),
    model_name text not null,
    rolling_log_loss double precision,
    weight double precision,               -- mean weight across digits, for quick queries
    weight_per_digit jsonb                 -- {"0": 0.12, "1": 0.09, ...} -- per-digit specialist weights
);
create index if not exists idx_astra_model_perf_symbol_model on astra_model_performance (symbol, model_name, ts desc);

create table if not exists astra_champion_challenger (
    id bigserial primary key,
    symbol text not null,
    ts timestamptz not null default now(),
    champion_log_loss double precision,
    challenger_log_loss double precision,
    improvement double precision,
    stable boolean,
    promoted boolean,
    weights jsonb
);

create table if not exists astra_experiment_log (
    id bigserial primary key,
    symbol text not null,
    ts timestamptz not null default now(),
    hypothesis text,
    metrics jsonb,
    decision text
);

create table if not exists astra_risk_events (
    id bigserial primary key,
    ts timestamptz not null default now(),
    symbol text,
    event_type text not null,              -- e.g. max_daily_loss_reached, emergency_stop
    detail jsonb
);

create table if not exists astra_system_events (
    id bigserial primary key,
    ts timestamptz not null default now(),
    component text not null,
    event_type text not null,
    detail jsonb
);

-- Restart-recovery snapshot: per-symbol digit history tail + learned
-- ensemble weights, so Astra doesn't have to relearn from zero after every
-- Railway restart. Overwritten (not appended) on a periodic cadence.
create table if not exists astra_symbol_state (
    symbol text primary key,
    updated_at timestamptz not null default now(),
    total_observed bigint not null default 0,
    recent_digits jsonb not null default '[]'::jsonb,   -- tail of the digit history (see config max_window)
    champion_weights jsonb,
    challenger_weights jsonb
);

-- Migration: adds the per-digit weight column to astra_model_performance
-- for deployments that ran `create table` before per-digit specialist
-- weighting was added. Safe to re-run.
alter table astra_model_performance add column if not exists weight_per_digit jsonb;

-- Migration: which architecture (global | specialist | hybrid) actually
-- produced a given prediction row. Safe to re-run.
alter table astra_predictions add column if not exists architecture text;

-- Architecture competition (see learning/architecture_competition.py):
-- rolling metric snapshots for each of the three competing architectures,
-- written whenever a promotion evaluation runs.
create table if not exists astra_architecture_performance (
    id bigserial primary key,
    symbol text not null,
    ts timestamptz not null default now(),
    architecture text not null,            -- global | specialist | hybrid
    composite_score double precision,
    log_loss double precision,
    brier double precision,
    calibration double precision,
    stability double precision,
    agreement double precision,
    economic_ev double precision,
    realized_pnl double precision,
    n integer
);
create index if not exists idx_astra_arch_perf_symbol_ts on astra_architecture_performance (symbol, ts desc);

-- Which architecture is currently the live champion per symbol (restart
-- recovery, same pattern as astra_symbol_state).
create table if not exists astra_architecture_state (
    symbol text primary key,
    champion_architecture text not null,
    updated_at timestamptz not null default now()
);

-- Restart-recovery snapshot for Architecture B's (models/digit_specialist.py)
-- 10 independent per-digit specialists. Without this, a Railway restart
-- resets every specialist back to its untrained Beta(1,9) prior and an
-- unfitted logistic regression, while Architecture A's champion_weights
-- already survive restarts via astra_symbol_state -- an unfair,
-- restart-triggered handicap in the architecture competition rather than a
-- reflection of which architecture actually predicts better. Overwritten
-- (not appended), same pattern as astra_symbol_state / astra_architecture_state.
create table if not exists astra_digit_specialist_state (
    symbol text primary key,
    updated_at timestamptz not null default now(),
    -- list of 10, one per digit 0-9, each:
    -- {alpha, beta, logistic: {coef, intercept, classes, t} | null}
    specialists jsonb not null
);

