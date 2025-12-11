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

## Phase 3: Agent Training
- [ ] **Setup Agent**
    - [ ] Configure PPO (SB3).
    - [ ] Set Hyperparameters (Batch size, Learning rate).
- [ ] **Training Loop**
    - [ ] Train on 2021 data.
    - [ ] Validate on 2022 data.
- [ ] **Evaluation**
    - [ ] Plot Cumulative Returns.
    - [ ] Calculate Sharpe Ratio.
