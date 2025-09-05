PY=python
ENV?=.venv
EXP?=churn-exp
DOCKER_IMAGE?=churn-mlops


init:
	python -m venv $(ENV) && . $(ENV)/bin/activate && pip install -U pip -r requirements.txt

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/train.py --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/evaluate.py --config configs/config.yaml

test:
	pytest -q

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-train:
	docker run --rm -e MLFLOW_EXPERIMENT_NAME=$(EXP) -v $(PWD)/data:/app/data -v $(PWD)/mlruns:/app/mlruns $(DOCKER_IMAGE)

docker-predict:
	docker run --rm -e MLFLOW_EXPERIMENT_NAME=$(EXP) -v $(PWD)/data:/app/data -v $(PWD)/mlruns:/app/mlruns $(DOCKER_IMAGE) python src/predict.py configs/config.yaml
