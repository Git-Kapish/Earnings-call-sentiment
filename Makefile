PY = .venv/Scripts/python
.PHONY: setup data score events analyze report test lint
setup:
	$(PY) -m pip install -r requirements.txt
data:
	PYTHONPATH=src $(PY) scripts/make_data.py
score:
	PYTHONPATH=src $(PY) scripts/make_scores.py
events:
	PYTHONPATH=src $(PY) scripts/make_events.py
analyze:
	PYTHONPATH=src $(PY) scripts/make_analysis.py
report:
	PYTHONPATH=src $(PY) scripts/make_report.py
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
