import os

import joblib
import pandas as pd


class PostprocessingPipeline:
    def __init__(self, config):
        self.cfg = config

    def run_train(self, model):
        """Save the trained model to disk."""
        model_path = self.cfg["pipeline_runner"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    def run_inference_static(self, y_pred):
        """Wrap the prediction(s) into a DataFrame for output."""
        if isinstance(y_pred, (list, tuple, pd.Series, pd.DataFrame)):
            val = y_pred[0] if len(y_pred) > 0 else None
        else:
            val = y_pred
        return pd.DataFrame({"prediction": [int(val)]})
