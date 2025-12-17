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

## Phase 6: Tuning & Benchmarking
- [x] **Hyperparameter Tuning (Optuna)**
    - [x] Create `tune_agent.py`.
    - [x] Define objective function (Maximize Sharpe/Reward).
    - [x] Optimize: `learning_rate`, `batch_size`, `n_steps`, `gamma`.
- [x] **Benchmarking**
    - [x] Create `benchmark_agent.py`.
    - [x] Implement Baselines: Buy & Hold (Equal Weight), Random Agent.
    - [x] Compare metrics (Cumulative Return, Sharpe, Max Drawdown).
- [x] **Documentation**
    - [x] Update `walkthrough.md` with Tuning results.

## Phase 7: Advanced Feature Engineering
- [x] **Macroeconomic Features**
    - [x] Import Interest Rate data (Fed, ECB, etc.) if available or proxy via Bond Yields (`^TNX`, `^DEZ`).
    - [x] Feature: Interest Rate Differential (Used `us_roi` as global risk factor).
- [x] **Microstructure Features**
    - [x] Add `ADX` (Average Directional Index) explicitly if not covered by `DX`.
    - [x] Add `ATR` (Average True Range) for volatility normalization.
- [x] **Data Expansion**
    - [x] Extend date range (e.g., 2018-2023) for better generalization.
    - [x] Add more pairs if relevant.
- [x] **Retrain & Benchmark**
    - [x] Train Agent on expanded/enhanced data.
    - [x] Run `tune_agent.py` again.
    - [x] Run `benchmark_agent.py` and aim for Positive Alpha (Achieved -0.27%, improved from -1.25%).

## Phase 8: User Experience & Parameterization
- [x] **CLI Improvements (Argparse)**
    - [x] Update `planning/run_pipeline.py` to accept `--tickers` and `--start-date`.
    - [x] Update `train_agent.py` to accept `--total-timesteps` and `--learning-rate`.
    - [x] Update `benchmark_agent.py` to accept `--model-path`.
- [x] **Notebook Enhancement**
    - [x] Update `alpha_fx_demo.ipynb` to invoke functions with parameters (e.g., `train(total_timesteps=50000)`).
    - [x] Demonstrate a "Higher Performance" run in the notebook.

## Phase 9: Visualization & Reporting
- [x] **Training Visualization**
    - [x] Update `train_agent.py` to use `Monitor` wrapper and save logs to `results/`.
    - [x] Add Notebook Cell: Plot Learning Curve (Reward vs Timesteps).
- [x] **Performance Visualization**
    - [x] Update `benchmark_agent.py` to save equity curves to `results/equity.csv`.
    - [x] Add Notebook Cell: Plot Equity Curve (Agent vs Baseline).
- [ ] **Tuning Visualization**
    - [x] (Optional) Visualize Optuna history if time permits.

## Phase 10: Advanced Parameterization
- [x] **Expose Hyperparameters**
    - [x] Update `train_agent.py` to accept:
        -   `lookback` (Window Size).
        -   `ent_coef` (Exploration Rate).
    - [x] Update CLI arguments.
- [x] **Notebook Updates**
    - [x] Update `alpha_fx_demo.ipynb` to explain and use these parameters.

## Phase 11: GPU Acceleration
- [x] **Device Selection**
    - [x] Update `train_agent.py` to accept `device` argument ('auto', 'cuda', 'cpu', 'mps').
    - [x] Update `alpha_fx_demo.ipynb` to demonstrate GPU usage.

## Phase 12: UX Improvements
- [x] **Progress Indicators**
    - [x] Update `train_agent.py` to use `progress_bar=True`.
    - [x] Clean up console output (reduce verbosity).

## Phase 13: Documentation & Explanations
- [x] **Ticker Explanation**
    - [x] Add currency pair details to `docs/usage.md`.
    - [x] Add currency pair details to `alpha_fx_demo.ipynb`.

## Phase 14: Interactive Visualization
- [x] **Interactive Chart**
    - [x] Create `interactive_plot.py` or similar logic.
    - [x] Inject interactive cell into `alpha_fx_demo.ipynb`.

## Phase 15: Visualization Refinement
- [x] **Dual-Axis Consolidation**
    - [x] Update plot logic to use `ax.twinx()` for secondary Y-axis.
    - [x] Consolidate Price and Indicator onto a single chart.
    - [x] Ensure clear legend for (L) and (R) axes.
