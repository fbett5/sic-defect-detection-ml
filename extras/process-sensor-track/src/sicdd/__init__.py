"""sicdd -- defect detection on in-line semiconductor process data.

A leakage-safe, reproducible pipeline for predicting end-of-line pass/fail
outcomes from in-line process sensor measurements (UCI SECOM), producing a
single portable model artifact that can be applied to new production data.
"""

__version__ = "0.1.0"

__all__ = ["data", "preprocess", "models", "evaluate", "train", "predict", "figures"]
