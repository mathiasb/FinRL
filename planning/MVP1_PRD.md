This is a high-potential project. Your hardware specifications (specifically the RTX 5070 Blackwell and M2 Ultra) are exceptional and actually exceed the requirements for a standard MVP, allowing us to bypass the cloud costs and latency of GCP for this initial phase.

[cite_start]The "Dependency Hell" described in the review [cite: 39] is the primary risk. Therefore, this brief prioritizes a **Local-First, Architectural remediation approach**. [cite_start]We will aim for **Strategy B (Unified Multi-Asset Portfolio Environment)** [cite: 88] rather than simple vectorization, as this offers the "meaningful" hedging capabilities necessary for FX.

Here is the Project Brief for **Project Alpha-FX**.

---

# Project Brief: FinRL-Gymnasium FX Integration (MVP)

## 1. Executive Summary
The goal is to engineer a robust, Gymnasium-native Foreign Exchange (FX) trading environment that resolves the current incompatibility between FinRL and Stable-Baselines3 (SB3) v2.0+. The MVP will demonstrate a Deep Reinforcement Learning (DRL) agent capable of managing a **multi-pair portfolio** (e.g., EURUSD, GBPUSD, USDJPY) with native handling of spreads and swap costs, trained on your local high-performance hardware.

## 2. Infrastructure & Deployment Strategy
**Recommendation: Hybrid Local Deployment**
We will forego GCP Vertex AI for the MVP. Your local hardware offers lower latency for the iterative debugging required to fix dependency issues.

* **Primary Training Node (Linux Server - RTX 5070):**
    * **Role:** Heavy lifting for Agent Training (PPO/SAC) and GPU-accelerated simulation (if using Podracer logic later).
    * **OS:** Ubuntu 22.04 LTS (Recommended for best CUDA stability).
    * **Config:** This machine will host the Docker container to ensure environment isolation from the host OS.
* **Development & Data Node (Mac Studio M2 Ultra):**
    * **Role:** Code development, DataOps (FinRL-Meta pipeline), and CPU-intensive backtesting (Vectorized Environments).
    * **Why:** The M2 Ultra's unified memory is excellent for handling large Pandas DataFrames during the feature engineering phase.

## 3. Technical Architecture (The "Validated Stack")
[cite_start]To avoid the "Dependency Hell" [cite: 39] [cite_start]and "NumPy 2.0 Incompatibility"[cite: 52], we will enforce the following strict versioning:

| Component | Version Constraint | Reason |
| :--- | :--- | :--- |
| **Python** | `3.10` | 3.11+ breaks legacy wheels; [cite_start]3.10 is the stability sweet spot[cite: 65]. |
| **NumPy** | `< 2.0.0` | [cite_start]**CRITICAL.** Prevents crash with Box2D/Legacy Gym types[cite: 56, 70]. |
| **Gymnasium** | `>= 0.29.0` | [cite_start]Enforces new API (`terminated`/`truncated`)[cite: 66]. |
| **Stable-Baselines3**| `>= 2.3.0` | [cite_start]Native Gymnasium backend support[cite: 67]. |
| **FinRL** | Source Install | [cite_start]Git clone required to get hotfixes not yet on PyPI[cite: 69]. |
| **Shimmy** | `>= 1.0.0` | [cite_start]Fallback bridge for legacy dependencies[cite: 68]. |

---

## 4. MVP Scope & Phased Execution

### Phase 1: DataOps & Feature Engineering (Days 1-3)
*Goal: Create a static, pre-processed dataset to decouple training from live data fetching issues.*
* **Input:** Minute-level OHLCV data for 3 major pairs (EURUSD, GBPUSD, USDJPY).
* **Process:** Use **FinRL-Meta** only for the extraction logic.
* **Engineering:**
    * [cite_start]Apply `FeatureEngineer` to add MACD, RSI, and Bollinger Bands[cite: 95].
    * [cite_start]**Crucial Step:** Normalize data using returns-based normalization or z-score standardization to handle the price difference between JPY pairs (~150.00) and USD pairs (~1.08)[cite: 96, 97].
* **Output:** A clean `.parquet` file (for speed) representing the "Tensor" of shape `(N_Pairs, Time, Features)`.

### Phase 2: The "Unified" Environment Construction (Days 4-7)
*Goal: Build `FXPortfolioEnv` that solves the single-pair constraint.*
[cite_start]Instead of using `gym-anytrading` (which is single-pair [cite: 72]), we will write a custom class inheriting from `gymnasium.Env`.
* **Action Space:** `Box(low=-1, high=1, shape=(3,))`. [cite_start]Represents portfolio weights (allowing long/short)[cite: 100].
* **Observation Space:** `Box(shape=(3, window_size, features))`.
* **Logic Upgrade:**
    * [cite_start]Implement `reset()` returning `(obs, info)` with seed support[cite: 115].
    * [cite_start]Implement `step()` returning `(obs, reward, terminated, truncated, info)`[cite: 125].
    * [cite_start]**Matrix Math:** Correctly implement the dot product of price changes $\times$ position vectors, accounting for lot sizes[cite: 103, 105].

### Phase 3: Agent Training (Days 8-10)
*Goal: Overfit a single batch, then train for generalization.*
* **Algorithm:** PPO (Proximal Policy Optimization) via SB3.
* **Hardware:** Run on the Linux/RTX 5070.
* [cite_start]**Validation:** Use strict "Rolling Window" validation (Train 2021, Test 2022) to prevent look-ahead bias[cite: 163].

---

## 5. Risk Mitigation Plan

* **Risk:** `ValueError: not enough values to unpack`.
    * *Mitigation:* This is the classic Gym vs. Gymnasium error. [cite_start]We will write a unit test specifically for the `env.step()` return signature before connecting the agent[cite: 49, 50].
* **Risk:** Bleeding Edge GPU (RTX 5070) Driver issues.
    * *Mitigation:* If CUDA 12.x/13.x issues arise with PyTorch, we will fall back to CPU training on the Mac Studio (slower, but safe) or use a Docker container with an older, stable CUDA image.
* **Risk:** Overfitting to noise.
    * [cite_start]*Mitigation:* We will penalize transaction costs heavily in the reward function to prevent the agent from "churning" (high-frequency trading that loses money on spreads)[cite: 26].

---

## 6. Next Step
To kick this off, we need to create the environment definition.

