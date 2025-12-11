# Project Alpha-FX Tasks

## Phase 1: DataOps & Feature Engineering
- [x] **Data Extraction (TDD)**
    - [x] Create `dataops` module structure.
    - [x] TDD: Implement `fetch_data` with mocked `yfinance`.
    - [x] TDD: Implement `clean_data` (handling missing values/NaNs).
    - [x] TDD: Implement `add_features` (technical indicators).
    - [x] TDD: Implement `normalize_data` (log returns).
    - [x] Integration: Run full pipeline to generate parquet.
    - [x] Resolution: Daily (1D) for 2021-2022.
    - [x] Pairs: EURUSD=X, GBPUSD=X, JPY=X, SEK=X.
- [ ] **Feature Engineering**
    - [ ] Apply `MACD`, `RSI`, `CCI`, `DX` (Standard FinRL set).
    - [ ] Add `Bollinger Bands`.
- [ ] **Normalization**
    - [ ] Implement Log Returns or Rolling Window scaling.
    - [ ] **Crucial**: Verify no look-ahead bias.
- [ ] **Output**
    - [ ] Save processed data to `data/fx_data_2021_2022.parquet`.

## Phase 2: Unified Environment Construction
- [x] **Scaffold Env Class (TDD)**
    - [x] Create `finrl/meta/env_fx_trading` directory.
    - [x] Create `tests/fixtures/fx_test_data.parquet` (Static data for consistent testing).
    - [x] TDD: Create `env_fx_portfolio.py` and `tests/test_env_fx_portfolio.py`.
    - [x] Inherit from `gymnasium.Env`.
- [x] **Define Spaces**
    - [x] Action Space: `Box` (Weights).
    - [x] Observation Space: `Box` (Windowed Features).
- [x] **Implement Logic**
    - [x] `reset()` with seed.
    - [x] `step()` with transaction costs and swap rates.
- [x] **Validation (TDD)**
    - [x] Test `reset()` seed consistency.
    - [x] Test `step()` invariant: Zero actions -> No change in portfolio value (minus costs).
    - [x] Test `step()` invariant: Costs are deducted correctly.
    - [x] Verify `check_env` from Stable-Baselines3.

## Phase 3: DevOps & Quality Assurance
- [x] **CI/CD Pipeline**
    - [x] Create `.github/workflows/ci_pr.yml`.
    - [x] Configure triggers: Push to main, PRs.
    - [x] Job: Linting (Ruff/Black).
    - [x] Job: Unit Tests (Pytest with Coverage).
- [x] **Development Standards**
    - [x] Create `docs/CONTRIBUTING.md` (TDD Guidelines).
    - [x] Create `docs/development_workflow.md`.
    - [x] Setup `pre-commit` hooks (Manual step managed by user/repo settings).

## Phase 4: Agent Training (TDD)
- [x] **Setup Agent Class**
    - [x] Create `finrl/agents/fx_agent.py`.
    - [x] TDD: Test Agent initialization (wraps PPO).
    - [x] Config: Hyperparameters.
- [x] **Training Loop**
    - [x] TDD: Test `train_model()` saves artifacts.
    - [x] Integration: Train on `fx_data_2021_2022.parquet`.
- [x] **Evaluation**
    - [x] TDD: Test `predict()` / `trade()`.
    - [x] Backtest on hold-out data (split from 2022).
    - [x] Calculate Sharpe Ratio.
    - [x] Backtest on hold-out data (split from 2022).
    - [x] Calculate Sharpe Ratio.

## Phase 5: Reward & State Engineering
- [x] **State Representation (TDD)**
    - [x] Define `get_state()` method in Env.
    - [x] TDD: Test state shape matches observation space `(n_tickers * window_size * n_features)`.
    - [x] Implement Windowing logic (Lookback).
- [x] **Reward Function (TDD)**
    - [x] Define `get_reward()` method in Env.
    - [x] TDD: Test Reward = Log Return * Weight.
    - [x] TDD: Test Risk-Adjusted Penalty (Volatility).
- [x] **Retrain & Verify**
    - [x] Retrain Agent with new Env logic.
    - [x] Compare Sharpe Ratio vs MVP baseline (MVP was NaN, now -2.28).
