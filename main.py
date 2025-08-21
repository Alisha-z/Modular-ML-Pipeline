# main.py
from pathlib import Path
import traceback
import yaml

from utils import get_logger
from preprocessing import run as preprocess_run
from train_model import run as train_run
from evaluate import run as evaluate_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logger = get_logger("main")
    try:
        # ensure standard folders exist
        for p in ["models", "data/processed", "outputs/figures", "outputs/reports", "logs"]:
            Path(p).mkdir(parents=True, exist_ok=True)

        logger.info("Loading config.yaml ...")
        config = load_config("config.yaml")

        logger.info("Step 1/3: Preprocessing ...")
        preprocess_run(config)  # saves preprocessor + splits via joblib

        logger.info("Step 2/3: Training ...")
        train_run(config)       # trains, saves best model

        logger.info("Step 3/3: Evaluation ...")
        evaluate_run(config)    # evaluates, saves metrics + plots

        logger.info("Pipeline finished successfully ✅")

    except Exception as e:
        logger.error("Pipeline failed ❌")
        logger.error(str(e))
        logger.debug(traceback.format_exc())
        raise
