.PHONY: setup run web test lint ingest seed demo clean

PY ?= python3

setup:  ## install dependencies
	$(PY) -m pip install -r requirements-dev.txt

run:  ## start the Streamlit app
	$(PY) -m streamlit run app.py

web:  ## start the v2 local web app (upload -> quiz -> report)
	$(PY) -m varkcheck.web --port 8765

test:  ## run the test suite (no network, no API key)
	$(PY) -m pytest -q

ingest:  ## (re)build the retrieval index
	$(PY) scripts/ingest.py --force

seed:  ## refresh the synthetic cohort
	$(PY) scripts/seed_demo.py --reset

demo:  ## end-to-end walkthrough without an API key
	$(PY) scripts/demo_flow.py

clean:  ## remove generated data and caches
	rm -rf data/index data/*.db data/*.db-wal data/*.db-shm logs __pycache__ .pytest_cache
