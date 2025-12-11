# System Architecture: DataOps & Environment

This document details the current architecture of the **DataOps Pipeline** and **FX Portfolio Environment** for the Alpha-FX project.

![System Architecture Overview](images/system_architecture_overview.png)

## 1. DataOps Module
The `dataops` module handles the extraction, transformation, and loading (ETL) of financial data.

### 1.1 FXDataPipeline Class
The core component is the `FXDataPipeline`, designed to be modular and testable.

```mermaid
classDiagram
    class FXDataPipeline {
        +__init__()
        +fetch_data(tickers: list, start: str, end: str, interval: str) pd.DataFrame
        +clean_data(df: pd.DataFrame) pd.DataFrame
        +add_features(df: pd.DataFrame, features: list) pd.DataFrame
        +normalize_data(df: pd.DataFrame) pd.DataFrame
    }
    note for FXDataPipeline "Handles ETL process.\nadd_features supports multi-ticker\ngrouping logic."
```

### 1.2 Data Processing Flow
The data flows through four distinct stages:

```mermaid
flowchart LR
    A["Raw Data (Yahoo)"] -->|fetch_data| B("DataFrame")
    B -->|clean_data| C{"Clean Data"}
    C -->|add_features| D["Technicals Added"]
    D -->|normalize_data| E["Log Returns / Scaled"]
    E -->|Save| F[("Parquet File")]
    
    subgraph Feature Engineering
    D -- "MACD, RSI, etc." --> D
    end
```

## 2. Environment Module
The `finrl.meta.env_fx_trading` module contains the Gymnasium-compliant trading environment.

### 2.1 FXPortfolioEnv Class
The environment simulates portfolio management with transaction costs and rebalancing logic.

```mermaid
classDiagram
    class GymEnv {
        <<Interface>>
        +reset()
        +step(action)
        +render()
    }
    
    class FXPortfolioEnv {
        +df : pd.DataFrame
        +initial_amount : float
        +state_space : int
        +action_space : Box
        +observation_space : Box
        
        +__init__(df, stock_dim, hmax, ...)
        +reset(seed=None) (np.array, dict)
        +step(actions: np.array) (np.array, float, bool, bool, dict)
    }
    
    FXPortfolioEnv --|> GymEnv : Inherits
```

### 2.2 Step Logic & State Transition
The `step()` method implements the core financial logic for state transitions.

```mermaid
sequenceDiagram
    participant Agent
    participant Env as FXPortfolioEnv
    participant Data as DataFrame
    
    Agent->>Env: step(action_weights)
    Note over Env: 1. Increment Day (t -> t+1)
    
    Env->>Data: Get prices/returns at t
    Data-->>Env: Returns
    
    Note over Env: 2. Calc Portfolio Value
    rect rgb(200, 255, 200)
    Note right of Env: V_new = V_old * (1 + w * ret)
    end
    
    Note over Env: 3. Rebalance & Costs
    rect rgb(255, 200, 200)
    Note right of Env: Turnover = |w_new - w_old|
    Note right of Env: V_net = V_new - Cost(Turnover)
    end
    
    Env-->>Agent: Returns (Obs, Reward, Done, Info)
```

## 3. Data Structures
### Observation Space
Currently defined as a Box of shape `(state_space,)`.
*Future Refinement*: `(n_tickers, window_size, features)`

### Action Space
Defined as a Box of shape `(n_tickers,)` representing portfolio weights.
*Constraint*: Weights should sum to <= 1.0 (remainder is cash).
