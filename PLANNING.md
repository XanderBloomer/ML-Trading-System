# ML Trading System — Project Planning Document

## Overview

An end-to-end **ML-powered quantitative research and signal generation platform**.
This is not just a trading bot — it is a full research system that pulls financial data,
engineers features, trains ML models, evaluates strategies via backtesting, and eventually
serves signals via an API.

---

## Goals

1. Build a production-grade ML system (not a script collection)
2. Learn quant finance fundamentals the right way (no data leakage, proper backtesting)
3. Create a platform that can evolve into a SaaS or personal trading system
4. Integrate LLMs for news-driven signal generation (later phase)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Dependency management | Poetry |
| Data | yfinance, pandas, numpy |
| ML | scikit-learn, XGBoost |
| Deep learning (later) | PyTorch |
| LLM (later) | OpenAI API |
| API | FastAPI + Uvicorn |
| Testing | pytest |
| Code quality | black, isort, flake8, pre-commit |
| Version control | Git + GitHub |
| Infra (later) | Docker, AWS |

---

## Repository Structure

```
ml-trading-system/
│
├── src/
│   └── ml_trading_system/
│       ├── data/           # Data ingestion and persistence
│       ├── features/       # Feature engineering
│       ├── models/         # ML model training and evaluation
│       ├── backtesting/    # Strategy backtesting engine
│       ├── llm/            # LLM integration (news, sentiment)
│       └── api/            # FastAPI application
│
├── tests/                  # Unit and integration tests
├── scripts/                # One-off runner scripts
├── configs/                # YAML config files (future)
├── notebooks/              # Exploration only (never production code)
├── data/                   # Local data store (gitignored)
│
├── pyproject.toml
├── PLANNING.md             # This file
└── README.md
```

---

## Development Phases

### ✅ Phase 1 — Foundations (COMPLETE)
**Goal:** Clean repo, working environment, data ingestion

- [x] Poetry environment set up
- [x] Full package structure created (`src/ml_trading_system/`)
- [x] `DataLoader` class — fetch, save, load OHLCV data from Yahoo Finance
- [x] `run_data_pipeline.py` — pulls AAPL, MSFT, SPY (1006 rows each, 2020–2024)
- [x] Code quality tooling — black, isort, flake8, pre-commit hooks
- [x] VS Code wired to Poetry environment
- [x] Tests passing

---

### ✅ Phase 2 — Feature Engineering (COMPLETE)
**Goal:** Transform raw OHLCV data into ML-ready features, strictly point-in-time

- [x] `FeatureEngineer` class built
- [x] 13 features implemented:
  - Returns: 1d, 5d, 10d, 21d, log return 1d
  - Volatility: rolling 5d, 21d
  - Momentum: price/SMA10, price/SMA50, SMA10/SMA50
  - RSI 14d
  - Volume: 1d change, volume ratio
- [x] Target variable: next-day return direction (binary: 1=up, 0=down)
- [x] No lookahead bias (target shifted correctly)
- [x] 8 tests passing

---

### 🔲 Phase 3 — First ML Model (NEXT)
**Goal:** Train, evaluate, and compare baseline models on real data

#### Tasks
- [ ] `ModelTrainer` class in `src/ml_trading_system/models/`
- [ ] Time-based train/test split (NOT random — critical for finance)
  - Train: 2020–2022
  - Test: 2023 (out-of-sample)
- [ ] Baseline model: Logistic Regression
- [ ] Second model: XGBoost classifier
- [ ] Evaluation metrics:
  - Accuracy
  - Precision / Recall
  - **Directional accuracy** (most important for trading)
  - Feature importance
- [ ] `run_model_training.py` script
- [ ] Tests for model pipeline

#### Key Principle
> Never use `train_test_split(shuffle=True)` on time-series data.
> Always split by date. Future data must never appear in training.

---

### 🔲 Phase 4 — Backtesting Engine
**Goal:** Evaluate trading strategies properly — this is the core edge of the system

#### Tasks
- [ ] `Backtester` class in `src/ml_trading_system/backtesting/`
- [ ] Simulate trades based on model signals
- [ ] Handle transaction costs (commission, slippage)
- [ ] Calculate performance metrics:
  - Total return
  - Sharpe ratio
  - Max drawdown
  - Win rate
- [ ] Compare strategy vs. buy-and-hold benchmark
- [ ] `run_backtest.py` script
- [ ] Tests for backtesting engine

#### Key Principle
> No lookahead bias in backtesting.
> Model can only use data available at the time of the signal.

---

### 🔲 Phase 5 — LLM Integration
**Goal:** Add news-driven signals using LLMs — this is what makes the project unique

#### Tasks
- [ ] News ingestion (financial news API or RSS feeds)
- [ ] LLM pipeline in `src/ml_trading_system/llm/`
- [ ] Per-article sentiment extraction:
  - Bullish / Bearish / Neutral
  - Confidence score
- [ ] Aggregate daily sentiment score per ticker
- [ ] Add sentiment as additional features to `FeatureEngineer`
- [ ] Re-train models with sentiment features
- [ ] Evaluate improvement in directional accuracy

#### Candidate APIs
- OpenAI GPT-4o (most capable)
- Groq (fast + cheap for high volume)
- Local model via Ollama (free, private)

---

### 🔲 Phase 6 — API Layer
**Goal:** Serve predictions and results via a clean REST API

#### Tasks
- [ ] FastAPI app in `src/ml_trading_system/api/`
- [ ] Endpoints:
  - `GET /signals/{ticker}` — latest model signal
  - `GET /backtest/{ticker}` — backtest results
  - `GET /health` — system health check
- [ ] Pydantic response models
- [ ] `run_api.py` script
- [ ] API tests

---

### 🔲 Phase 7 — Dashboard (Optional)
**Goal:** Visualise signals, PnL, and model outputs

- [ ] Simple web dashboard (Streamlit for speed, or React later)
- [ ] Show: signals, portfolio PnL, model confidence, news sentiment

---

### 🔲 Phase 8 — Advanced (Future)
- [ ] Multi-asset portfolio optimisation
- [ ] Risk management module (position sizing, stop-loss)
- [ ] Live paper trading (Alpaca API)
- [ ] Docker containerisation
- [ ] AWS deployment (ECS + S3)
- [ ] CI/CD via GitHub Actions

---

## Core Engineering Principles

1. **No lookahead bias** — the single most important rule in quant ML
2. **Time-based splits only** — never shuffle time-series data
3. **Separation of concerns** — data / features / models / backtest are fully decoupled
4. **Reproducibility** — every result must be reproducible from a clean clone
5. **Test everything** — especially data pipelines and feature calculations
6. **Config-driven** — avoid hardcoded values; use config files for parameters

---

## Current State (as of last session)

| Component | Status | File |
|---|---|---|
| DataLoader | ✅ Complete | `src/ml_trading_system/data/data_loader.py` |
| FeatureEngineer | ✅ Complete | `src/ml_trading_system/features/feature_engineer.py` |
| run_data_pipeline | ✅ Complete | `scripts/run_data_pipeline.py` |
| Tests | ✅ 8 passing | `tests/` |
| Model training | 🔲 Not started | `src/ml_trading_system/models/` |
| Backtesting | 🔲 Not started | `src/ml_trading_system/backtesting/` |
| LLM integration | 🔲 Not started | `src/ml_trading_system/llm/` |
| API | 🔲 Not started | `src/ml_trading_system/api/` |

---

## How to Run (from repo root)

```bash
# Install dependencies
poetry install

# Fetch market data
poetry run python scripts/run_data_pipeline.py

# Run tests
poetry run pytest

# Run linting
poetry run flake8 src/ tests/
```

---

## Next Immediate Task

**Phase 3: Model Training**

Build `src/ml_trading_system/models/trainer.py`:
- Time-based train/test split
- Logistic Regression baseline
- XGBoost model
- Evaluation metrics including directional accuracy