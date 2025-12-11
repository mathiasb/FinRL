# FinRL Alpha-FX Usage Guide

This guide provides a step-by-step tutorial on how to use the Alpha-FX system, from data preparation to agent training and evaluation.

## 1. Setup

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone <your-repo-url>
cd FinRL

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 2. Data Preparation (DataOps)

Before training, you need to download and process the FX data. The pipeline fetches data from Yahoo Finance, cleans it, adds technical indicators, and saves it as a Parquet file.

**Features Generated:**
*   **MACD**: Moving Average Convergence Divergence (default `stockstats` settings: fast=12, slow=26, sign=9).
*   **RSI**: Relative Strength Index (30-day window).
*   **CCI**: Commodity Channel Index (30-day window).
*   **DX**: Directional Movement Index (30-day window).
*   **Bollinger Bands**: Upper (`boll_ub`) and Lower (`boll_lb`) bands.

**Run the pipeline script:**

```bash
python planning/run_pipeline.py
```

*   **Input**: Fetches data for `EURUSD=X`, `GBPUSD=X`, `JPY=X`, `SEK=X`, and `EURSEK=X`. (Configured in script: `planning/run_pipeline.py`).
*   **Output**: Creates `data/fx_data_2021_2022.parquet`.

## 3. Training the Agent

Once the data is ready, you can train the Deep Reinforcement Learning agent.

**Algorithm: PPO (Proximal Policy Optimization)**
We use **PPO**, a policy gradient method that alternates between sampling data through interaction with the environment and optimizing a "surrogate" objective function using stochastic gradient ascent. It is chosen for its stability, ease of tuning, and sample efficiency compared to other on-policy methods.

**Run the training script:**

```bash
python train_agent.py
```

*   **Process**:
    *   Loads `data/fx_data_2021_2022.parquet`.
    *   Initializes the `FXPortfolioEnv` with transaction costs (0.1%).
    *   Trains a PPO agent for 5,000 timesteps (default MVP setting).
    *   Saves the trained model to `models/fx_agent_mvp.zip`.

**Interpreting Training Output:**
When the script runs, it logs metrics to the console. Key metrics to watch:
*   `ep_rew_mean` (Episode Reward Mean): The average reward per episode. This should **increase** over time, indicating the agent is learning a profitable strategy.
*   `fps` (Frames Per Second): Speed of training.
*   `explained_variance`: How well the Value Function predicts returns. Values close to **1.0** are ideal. Low or negative values indicate the value function is struggling.
*   `loss`: The PPO loss. This may fluctuate but shouldn't explode.

*Tip: If `ep_rew_mean` is flat or negative, try adjusting hyperparameters or checking if your data contains valid signals.*

## 4. Evaluation & Backtesting

After training, evaluate the agent's performance on unseen data (or the 2022 subset).

**Run the backtest script:**

```bash
python backtest_agent.py
```

*   **Process**:
    *   Loads the saved model from `models/fx_agent_mvp.zip`.
    *   Runs the agent on data from 2022-01-01 onwards.
    *   Calculates performance metrics (Sharpe Ratio, Final Portfolio Value).

**Interpreting Backtest Results:**
The script prints the Final Portfolio Value and Sharpe Ratio.
*   **Final Portfolio Value**: Compare this to your `initial_amount` (e.g., 1,000,000). A value > Initial indicates profit.
*   **Sharpe Ratio**: A measure of risk-adjusted return.
    *   `> 1.0`: Good.
    *   `> 2.0`: Excellent.
    *   `< 0`: The strategy is losing money or taking excessive risk for the return.
*   **Sanity Check**: If the agent makes huge profits in training but loses in backtesting, it is likely **overfitting**. Try reducing model complexity or adding regularization.

## 5. Development Workflow

For developers contributing to the project, please refer to:
*   [Development Workflow](development_workflow.md): TDD guidelines and CI/CD details.
*   [System Architecture](system_architecture.md): Detailed UML diagrams of the system.
