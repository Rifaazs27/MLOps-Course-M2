# tests/test_pipeline.py
from src.pipeline import build_pipeline

def test_build_pipeline():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    assert "pre" in dict(pipe.named_steps)
    assert "model" in dict(pipe.named_steps)
