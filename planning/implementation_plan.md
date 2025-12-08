# Implementation Plan - Phase 1: DataOps

## Goal
Create a robust data pipeline to fetch, process, and save FX data for training.

## User Review Required
> [!WARNING]
> **Data Resolution**: Yahoo Finance **does not support** minute-level data for 2021 (limit is 7-30 days).
> **Proposal**: I will use **Daily (1D)** data for the MVP to ensure we can get the full 2021-2022 range.
> *Alternative*: If you have a premium API key (Alpaca/Polygon) or a local `.csv` file with minute data, please provide it.

## Proposed Changes

### [New Script] `planning/phase1_dataops.py`
This script will:
1.  **Define Tickers**: `EURUSD=X`, `GBPUSD=X`, `JPY=X`, `SEK=X`.
2.  **Fetch Data**: Use `FinRL.meta.data_processors.processor_yahoofinance.YahooFinanceProcessor`.
3.  **Feature Engineering**:
    - Use `StockDataFrame` (stockstats) via FinRL's `add_technical_indicator`.
    - Add: `macd`, `rsi_30`, `cci_30`, `dx_30`, `boll_ub`, `boll_lb`.
4.  **Normalization**:
    - Compute **Log Returns**: `np.log(close / close.shift(1))`.
    - Drop `NaN`s created by shifting.
5.  **Save**: Export to `data/fx_data_2021_2022.parquet`.

## Verification Plan
### Automated Tests
- Run `python planning/phase1_dataops.py`.
- Check if `data/fx_data_2021_2022.parquet` exists.
- Load the parquet file and print `df.head()` and `df.shape` to verify columns and rows.
