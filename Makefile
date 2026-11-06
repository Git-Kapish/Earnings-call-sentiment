PY = .venv/Scripts/python
.PHONY: setup data score events analyze report test lint
setup:
	$(PY) -m pip install -r requirements.txt
data:
	$(PY) scripts/make_data.py
score:
	$(PY) scripts/make_scores.py
events:
	$(PY) scripts/make_events.py
analyze:
	$(PY) scripts/make_analysis.py
report:
	$(PY) scripts/make_report.py
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
