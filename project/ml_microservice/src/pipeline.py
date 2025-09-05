# src/pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(numeric, categorical, model_type="logreg"):
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop"
    )
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Important: keep the name "model" (tests s'attendent à "model")
    return Pipeline(steps=[("pre", pre), ("model", model)])
