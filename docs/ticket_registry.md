# Ticket Registry

| ID     | Title                          | Status | Files                         | Notes |
|--------|--------------------------------|--------|-------------------------------|-------|
| RL-001 | Project bootstrap              | Done   | structure, config, docs       |       |
| RL-002 | Data ingestion layer           | Done   | src/data/, src/config.py, main.py |  |
| RL-003 | Environment & reproducibility  | Done   | requirements.txt, .gitignore, README.md, Makefile | |
| RL-004 | No-accumulation data policy    | Done   | src/data/storage.py, main.py, tests/ | |
| RL-005 | Documentation cleanup          | Done   | docs/                         | Simplified workflow, removed prompts registry |
| RL-006 | Historical risk analytics      | Done   | src/analytics/risk_metrics.py | Series-in API, pure functions |
| RL-007 | Monte Carlo + analysis notebook | Done   | src/analytics/monte_carlo.py, notebooks/ | GBM, VaR/CVaR |
| RL-008 | Streamlit app (full UI)         | Done   | app/streamlit_app.py          | Pure integration, no logic |
