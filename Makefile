.PHONY: setup activate install test verify run app clean

setup: ## Create virtual environment
	python -m venv .venv

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run test suite
	pytest -v

verify: ## Verify environment is working
	python -c "import pandas; import numpy; import yfinance; print('Environment OK')"

run: ## Run CLI data ingestion (usage: make run TICKER=BTC-USD)
	python main.py --ticker $(TICKER)

app: ## Launch Streamlit app
	streamlit run app/streamlit_app.py

clean: ## Remove cached files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
