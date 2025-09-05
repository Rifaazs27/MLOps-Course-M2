# src/evaluate.py
import argparse
import os
import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from src.utils import load_config, load_data, split_data, save_fig

mlflow.set_experiment("ChurnClassifier")

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model-path", default="models/best_model.joblib")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    df = load_data(cfg["data"]["csv_path"])
    target = cfg["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target].map({"Yes":1, "No":0}) if df[target].dtype == object else df[target]

    _, X_test, _, y_test = split_data(X, y,
                                     test_size=cfg["data"]["test_size"],
                                     random_state=cfg["data"]["random_state"])

    # load model
    model = joblib.load(args.model_path)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    roc_path = "plots/roc.png"
    save_fig(fig_roc, roc_path)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    fig_pr = plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend()
    pr_path = "plots/pr.png"
    save_fig(fig_pr, pr_path)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp_fig = plt.figure()
    disp = ConfusionMatrixDisplay(cm).plot()
    cm_path = "plots/confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # Log artifacts to MLflow
    mlflow.log_artifact(roc_path, artifact_path="plots")
    mlflow.log_artifact(pr_path, artifact_path="plots")
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # Save classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    import json
    with open("outputs/classification_report.json", "w") as f:
        json.dump(report, f)
    mlflow.log_artifact("outputs/classification_report.json", artifact_path="outputs")

if __name__ == "__main__":
    main()
