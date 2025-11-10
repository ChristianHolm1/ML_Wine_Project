# ===============================================================
# postprocessing.py
# ---------------------------------------------------------------
# TRAINING: save the model
# INFERENCE: format the prediction row (optionally as integer)
# ===============================================================

from typing import Dict, Any
import os
import math
import joblib
import pandas as pd


class PostprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # ----------------------------- TRAINING -----------------------------
    def run_train(self, model) -> None:
        """Persist the trained model to the configured path."""
        model_path = self.cfg["pipeline_runner"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    def run_inference_static(self, y_pred: float) -> pd.DataFrame:

        infer = self.cfg.get("inference", {})
        fmt = infer.get("format", {})
        as_int = bool(fmt.get("as_integer", False))
        strategy = fmt.get("integer_strategy", "round")
        min_value = float(fmt.get("min_value", 0.0))

        val = float(y_pred)
        val = max(min_value, val)

        import math
        if as_int:
            if strategy == "floor":
                val = math.floor(val)
            elif strategy == "ceil":
                val = math.ceil(val)
            else:
                val = int(round(val))

        return pd.DataFrame({"prediction": [val]})
