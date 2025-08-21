# utils.py
import logging

def get_logger(name: str = __name__):  # Accept name parameter
    logger = logging.getLogger(name)  # Use the name here
    logger.setLevel(logging.INFO)  # Set level on the logger instance

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # Include name in format
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def run(config):
    # Access the problem type from the config
    try:
        problem_type = config["problem_type"]
    except KeyError as e:
        raise KeyError(f"Missing key in config: {e}")

    # Continue with the training logic based on the problem type
    if problem_type == "classification":
        # Training logic for classification
        pass
    elif problem_type == "regression":
        # Training logic for regression
        pass
    else:
        raise ValueError("Invalid problem type specified in config.")

