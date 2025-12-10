# Implementation Plan - Phase 1: DataOps (TDD)

## Goal
Implement a robust, tested data pipeline for FX data using TDD. The pipeline will handle fetching, cleaning, feature engineering, and normalization.

## User Review Required
> [!NOTE]
> This plan replaces the previous single-script approach. We are now building a modular `dataops` package.

## Proposed Changes

### [New Module] `dataops`

#### [NEW] [pipeline.py](file:///Users/mathias/Documents/local-dev/AI/FinRL/dataops/pipeline.py)
This file will contain the `FXDataPipeline` class with methods:
- `fetch_data(tickers, start_date, end_date, interval)`: Wraps `yfinance.download`.
- `clean_data(df)`: Handles missing values and formatting.
- `add_features(df)`: Adds technical indicators (MACD, RSI, etc.).
- `normalize_data(df)`: Computes log returns.

#### [NEW] [tests/test_pipeline.py](file:///Users/mathias/Documents/local-dev/AI/FinRL/dataops/tests/test_pipeline.py)
Unit tests for the pipeline.
- `test_fetch_data_calls_yfinance_correctly` (Implemented, Failing)
- `test_clean_data_handles_nans` (Planned)
- `test_add_features_adds_columns` (Planned)
- `test_normalize_computes_log_returns` (Planned)

## Verification Plan

### Automated Tests
Run the specific test suite:
```bash
.venv/bin/python -m unittest dataops/tests/test_pipeline.py
```

### Manual Verification
After all tests pass, we will run a designated "integration" script (or a main block in `pipeline.py`) to actually fetch data and inspect the output file `data/fx_data_2021_2022.parquet`.

