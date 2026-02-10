# Gold Trading AI - Makefile
# ===========================

.PHONY: install run train app clean help

# Python executable
PYTHON = python

help:
	@echo "Gold Trading AI - Available Commands"
	@echo "====================================="
	@echo "make install  - Install dependencies"
	@echo "make run      - Run complete pipeline"
	@echo "make train    - Train model only"
	@echo "make app      - Launch Streamlit app"
	@echo "make clean    - Clean cache files"

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) run_pipeline.py

train:
	$(PYTHON) -c "from model_training import train_and_save_model; from feature_engineering import process_raw_data; train_and_save_model(process_raw_data())"

app:
	$(PYTHON) -m streamlit run app.py

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	rm -rf logs/*

data:
	$(PYTHON) -c "from data_collector import collect_and_save_data; collect_and_save_data()"
