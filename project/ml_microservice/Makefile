PY=python
ENV?=.venv
EXP?=churn-exp

init:
	$(PY) -m venv $(ENV)
	. $(ENV)/bin/activate && pip install -U pip -r requirements.txt

clean-data:
	$(PY) configs/run-clean-telco.py

train:
	export MLFLOW_EXPERIMENT_NAME=$(EXP) && export PYTHONPATH=. && \
	$(PY) src/train.py --config configs/config.yaml

evaluate:
	export MLFLOW_EXPERIMENT_NAME=$(EXP) && export PYTHONPATH=. && \
	$(PY) src/evaluate.py --config configs/config.yaml

test:
	PYTHONPATH=. pytest -q

mlflow-ui:
	mlflow ui --port 5005

all: clean-data train evaluate mlflow-ui


