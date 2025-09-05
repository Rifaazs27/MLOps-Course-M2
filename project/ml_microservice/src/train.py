# src/train.py
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import logging

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import pandas as pd

from src.pipeline import build_pipeline
from src.utils import load_config, load_data, split_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def ensure_list_params(params):
    return {k: (v if isinstance(v, list) else [v]) for k, v in params.items()}

def main():
    args = parse_args()
    cfg = load_config(args.config)

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "churn-exp"))
    mlflow.sklearn.autolog()

    df = load_data(cfg["data"]["csv_path"])
    target = cfg["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target].map({"Yes":1, "No":0}) if df[target].dtype == object else df[target]

    X_train, X_test, y_train, y_test = split_data(X, y,
                                                 test_size=cfg["data"]["test_size"],
                                                 random_state=cfg["data"]["random_state"])

    pipe = build_pipeline(cfg["features"]["numeric"], cfg["features"]["categorical"], model_type=cfg["model"]["type"])

    # convert model params to pipeline param names
    params = ensure_list_params(cfg["model"].get("params", {}))
    param_grid = {f"model__{k}": v for k, v in params.items()}

    cv_cfg = cfg.get("cv", {})
    if cv_cfg.get("strategy") == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=cv_cfg.get("n_splits", 5), shuffle=True, random_state=cfg["data"]["random_state"])
    else:
        cv = 5

    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=cv_cfg.get("scoring", "roc_auc"), n_jobs=-1, verbose=1)
    logger.info("Lancement de l'entraînement (GridSearchCV)...")
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    logger.info(f"Best params: {gs.best_params_}")
    # Eval on test
    if hasattr(best, "predict_proba"):
        y_proba = best.predict_proba(X_test)[:, 1]
    else:
        # fallback
        y_proba = best.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("test_roc_auc", auc)
    mlflow.log_metric("test_accuracy", acc)
    logger.info(f"Test ROC AUC: {auc:.4f}  Accuracy: {acc:.4f}")

    # Save model locally and log as artifact
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_model.joblib")
    joblib.dump(best, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    # Option: register model if requested
    registered_name = cfg["model"].get("registered_name")
    try:
        if registered_name:
            mlflow.sklearn.log_model(best, artifact_path="model", registered_model_name=registered_name)
            logger.info(f"Model logged and registered as '{registered_name}'")
    except Exception as e:
        logger.warning(f"Model registration failed (ok on local file store): {e}")

    # Save predictions CSV for error analysis
    preds_df = pd.DataFrame({
        "y_true": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred
    }, index=X_test.index)
    preds_csv = "outputs/predictions.csv"
    os.makedirs(os.path.dirname(preds_csv), exist_ok=True)
    preds_df.to_csv(preds_csv, index=True)
    mlflow.log_artifact(preds_csv, artifact_path="outputs")

if __name__ == "__main__":
    main()
