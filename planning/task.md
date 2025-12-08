# Project Alpha-FX Tasks

## Phase 1: DataOps & Feature Engineering
- [x] **Data Extraction**
    - [x] Create script `planning/phase1_dataops.py` to fetch data.
    - [ ] Resolution: Daily (1D) for 2021-2022 (Due to Yahoo Finance 1m limitation).
    - [ ] Pairs: EURUSD=X, GBPUSD=X, JPY=X, SEK=X (Yahoo tickers).
- [ ] **Feature Engineering**
    - [ ] Apply `MACD`, `RSI`, `CCI`, `DX` (Standard FinRL set).
    - [ ] Add `Bollinger Bands`.
- [ ] **Normalization**
    - [ ] Implement Log Returns or Rolling Window scaling.
    - [ ] **Crucial**: Verify no look-ahead bias.
- [ ] **Output**
    - [ ] Save processed data to `data/fx_data_2021_2022.parquet`.

## Phase 2: Unified Environment Construction
- [ ] **Scaffold Env Class**
    - [ ] Create `finrl/meta/env_fx_trading/env_fx_portfolio.py`.
    - [ ] Inherit from `gymnasium.Env`.
- [ ] **Define Spaces**
    - [ ] Action Space: `Box` (Weights).
    - [ ] Observation Space: `Box` (Windowed Features).
- [ ] **Implement Logic**
    - [ ] `reset()` with seed.
    - [ ] `step()` with transaction costs and swap rates.
- [ ] **Validation**
    - [ ] Unit test `env.step()` signature.
    - [ ] Verify `check_env` from Stable-Baselines3.

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
