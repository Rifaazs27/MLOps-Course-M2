# src/utils.py
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Example: Telco dataset has 'TotalCharges' sometimes as string -> coerce
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Convert target to 0/1 if it's "Yes"/"No"
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
