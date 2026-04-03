# RiskLens — Investor Risk Assistant

Decision-support app for contextual financial asset risk assessment.

## Setup

### 1. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Mac / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

**CLI (data ingestion):**

```bash
python main.py --ticker BTC-USD
```

**Streamlit app:**

```bash
streamlit run app/streamlit_app.py
```

### 4. Verify setup

```bash
python -c "import pandas; import numpy; import yfinance; print('Environment OK')"
pytest --co -q
```

## Project Structure

```
risklens/
├── app/                  # Streamlit frontend
├── src/                  # Core source code
│   ├── data/             # Data fetch, process, validate
│   ├── analytics/        # Risk metrics and models
│   ├── pipelines/        # Orchestration
│   ├── services/         # External integrations
│   ├── domain/           # Domain objects
│   └── utils/            # Shared utilities
├── data/
│   ├── raw/              # Raw downloaded data (gitignored)
│   ├── interim/          # Intermediate artifacts (gitignored)
│   └── processed/        # Clean output data (gitignored)
├── docs/                 # Project documentation
├── notebooks/            # Exploratory notebooks
├── tests/                # Test suite
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
└── .gitignore
```
