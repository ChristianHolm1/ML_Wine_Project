# ===============================================================
# PipelineRunner — TRAINING + INFERENCE orchestration (hybrid)
# Path: ml/src/pipelines/pipeline_runner.py
# ===============================================================
from typing import Dict, Any, List, Union
import pandas as pd

from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.training import TrainingPipeline
from pipelines.postprocessing import PostprocessingPipeline
from pipelines.inference import InferencePipeline


class PipelineRunner:
    """
    Orchestrates both time-series and non–time-series workflows.

    TIME-SERIES MODE (streaming or batch stepping):
      • TRAINING:   raw → preprocess → FE → train → save model
      • INFERENCE:  append new row by timestamp → last-N → preprocess → FE → predict t+1
                     → save prediction (shifted time) → persist DB → backfill actuals

    STATIC/NON–TIME-SERIES MODE:
      • TRAINING:   dataset (DataFrame) → preprocess → FE → train → save model
      • INFERENCE:  DataFrame or list[dict] → preprocess → FE → predict per-row → save
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        self.cfg = config
        self.dm = data_manager

        # Shared steps
        self.prep = PreprocessingPipeline(config)
        self.post = PostprocessingPipeline(config)

        # === TRAINING ===
        self.train = TrainingPipeline(config)

        # === INFERENCE ===
        self.inf = InferencePipeline(config)

        # Optional in-memory dataset for static mode
        self.dataset: Union[pd.DataFrame, None] = None

    # ============================== TRAINING (static) ==============================
    def run_training_static(self, df: pd.DataFrame) -> None:
        """
        Train using an *in-memory* DataFrame (tabular dataset).
        Intended for non-time-series datasets (e.g. wine).
        This will run preprocessing, (optionally) feature engineering,
        training, and postprocessing.save/model.
        """
        if df is None or df.empty:
            raise ValueError("[training] Provided dataframe is empty.")

        # 1) Preprocess
        df = self.prep.run(df)
        if df is None or df.empty:
            raise ValueError("[training] Dataframe empty after preprocessing.")

        # 3) Train
        model = self.train.run(df)

        # 4) Postprocessing (save model, etc.)
        self.post.run_train(model)

    # ============================== INFERENCE (static single) ==============================
    def predict_from_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict for a single NON–time-series record.
        """
        df = pd.DataFrame([values])
        df = self.prep.run(df)

        target_col = self.cfg["training"]["target"].get("source_col")
        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])

        y_hat = self.inf.run(df)
        df_pred = self.post.run_inference_static(y_pred=y_hat)

        pred_val = df_pred["prediction"].iloc[0]
        return {
            "input": values,
            "prediction": int(pred_val) if float(pred_val).is_integer() else float(pred_val),
        }
