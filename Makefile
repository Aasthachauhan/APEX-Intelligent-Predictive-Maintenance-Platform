.PHONY: install train test lint serve smoke
install: ; pip install -r requirements.txt ruff mypy bandit pytest-cov pandas-stubs
train:   ; PYTHONPATH=. python -m ml.training.train --mlflow
test:    ; PYTHONPATH=. pytest --cov=backend/app --cov=ml
lint:    ; ruff check . && ruff format --check . && mypy ml backend/app && bandit -q -r ml backend/app
serve:   ; PYTHONPATH=. uvicorn backend.app.main:app --reload
smoke:   ; python scripts/smoke_e2e.py
