from __future__ import annotations
import importlib
from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV
from utils import get_logger  # <-- FIXED: absolute import (no leading dot)


def _make_estimator(qualified_name: str):
    module_name, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def default_scoring(problem_type: str):
    if problem_type == "classification":
        return "accuracy"
    return "neg_root_mean_squared_error"  # for regression

def run(config):
        # Access the problem type from the config
        problem_type = config.get("problem_type")

        if problem_type == "classification":
            try:
                models = config["classification_models"]
            except KeyError as e:
                raise KeyError(f"Missing key in config: {e}")

            # Continue with the training logic for classification models
            for model in models:
                model_name = model["name"]
                model_params = model["params"]
                # Train the model using model_name and model_params
                pass
        elif problem_type == "regression":
            try:
                models = config["regression_models"]
            except KeyError as e:
                raise KeyError(f"Missing key in config: {e}")

            # Continue with the training logic for regression models
            for model in models:
                model_name = model["name"]
                model_params = model["params"]
                # Train the model using model_name and model_params
                pass
        else:
            raise ValueError("Invalid problem type specified in config.")

