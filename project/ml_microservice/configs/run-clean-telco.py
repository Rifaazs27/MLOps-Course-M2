import pandas as pd
import os

# chemin relatif au projet, peu importe où on exécute le script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_path = os.path.join(base_dir, "data", "data.csv")
output_path = os.path.join(base_dir, "data", "raw.csv")

df = pd.read_csv(input_path)

# nettoyage spécifique Telco
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# sauvegarde
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved cleaned data to {output_path} — shape: {df.shape}")
