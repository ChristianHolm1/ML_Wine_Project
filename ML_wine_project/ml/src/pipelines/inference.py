# ===============================================================
# InferencePipeline — INFERENCE ONLY (pipeline module)
# ===============================================================
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class InferencePipeline:
    """
    Loads the trained model and returns the LAST prediction from a prepared batch.

    Robustness steps:
      1) Align columns to the model's training schema (feature_names_in_)
      2) ffill/bfill remaining NaNs using only the batch history
      3) Drop any still-NaN rows
      4) Predict and map factorized indices back to original labels if mapping exists
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        cfg: optional config dict. If provided, may contain:
            - artifacts.model_path: path to saved pipeline (joblib)
        """
        self.cfg = cfg or {}
        self.pipeline_path = self.cfg.get("artifacts", {}).get("model_path", "models/pipeline.joblib")
        self.pipeline = None

    def load_pipeline(self, path: Optional[str] = None):
        """
        Load pipeline from disk into self.pipeline. If path argument provided, use it.
        """
        if path is None:
            path = self.pipeline_path

        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Pipeline artifact not found at: {path}")

        try:
            self.pipeline = joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline from {path}: {e}")

        return self.pipeline

    def _get_model_and_clf(self):
        """
        Returns (pipeline, clf) where clf is the final estimator in a sklearn pipeline
        or the pipeline itself if it's not a pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first or set self.pipeline.")

        # try to extract final estimator (named_steps)
        clf = None
        try:
            if hasattr(self.pipeline, "named_steps"):
                # typical names; if none match, take last step
                for candidate in ("clf", "classifier", "estimator", "model"):
                    if candidate in self.pipeline.named_steps:
                        clf = self.pipeline.named_steps[candidate]
                        break
                if clf is None:
                    # take last step by position
                    steps = list(self.pipeline.named_steps.values())
                    if steps:
                        clf = steps[-1]
            else:
                # pipeline may be a raw estimator
                clf = self.pipeline
        except Exception:
            clf = self.pipeline

        return self.pipeline, clf

    def _align_columns(self, x: pd.DataFrame, pipeline):
        """
        Align incoming dataframe columns to what the pipeline expects.
        If pipeline has feature_names_in_ (scikit-learn 1.0+), prefer that.
        Otherwise, attempt to use pipeline.named_steps['scaler'] or leave as-is.
        """
        expected = None
        try:
            # Many sklearn transformers/estimators expose feature_names_in_ on fitted estimators
            if hasattr(pipeline, "feature_names_in_"):
                expected = list(pipeline.feature_names_in_)
        except Exception:
            expected = None

        # If pipeline is a Pipeline with a scaler or first transformer that has feature_names_in_:
        if expected is None and hasattr(pipeline, "named_steps"):
            for step in pipeline.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    expected = list(step.feature_names_in_)
                    break

        # If we still don't know expected, just use the incoming columns
        if expected is None:
            expected = list(x.columns)

        # Keep only expected columns and preserve order
        missing = [c for c in expected if c not in x.columns]
        if missing:
            # We'll keep alignment and allow later ffill/bfill to fill batch history values.
            # Create missing columns with NaN so alignment is explicit
            for c in missing:
                x[c] = np.nan

        x_aligned = x.loc[:, expected].copy()
        return x_aligned

    def _map_predictions_to_original(self, raw_preds, pipeline, clf):
        """
        Given raw_preds (1D array-like) from pipeline.predict or argmax(predict_proba),
        try to map integer indices back to the original labels in this order:
          1) pipeline.label_mapping_ (np.ndarray)
          2) pipeline.label_encoder_.inverse_transform(...)
          3) clf.classes_ mapping (if present and raw_preds are integer indices)
          4) If raw_preds already appear to be original labels, return them unchanged
          5) Last resort: return raw_preds unchanged (but caller should be warned)
        Returns mapped_preds (numpy array) and a small string description of which mapping was used.
        """
        raw = np.asarray(raw_preds)
        mapped = None
        reason = "none"

        # 1) pipeline.label_mapping_ (pd.Index or np.ndarray)
        if hasattr(pipeline, "label_mapping_"):
            try:
                mapping = np.asarray(getattr(pipeline, "label_mapping_"))
                if mapping.size > 0 and np.issubdtype(raw.dtype, np.integer):
                    if raw.min() >= 0 and raw.max() <= (len(mapping) - 1):
                        mapped = mapping[raw.astype(int)]
                        reason = "pipeline.label_mapping_"
                        return np.asarray(mapped), reason
            except Exception:
                # fall through to next option
                pass

        # 2) label_encoder_
        if hasattr(pipeline, "label_encoder_"):
            try:
                le = getattr(pipeline, "label_encoder_")
                # label encoder expects integer indices
                if np.issubdtype(raw.dtype, np.integer):
                    inv = le.inverse_transform(raw.astype(int))
                    mapped = np.asarray(inv)
                    reason = "pipeline.label_encoder_.inverse_transform"
                    return mapped, reason
            except Exception:
                pass

        # 3) classifier.classes_
        if clf is not None and hasattr(clf, "classes_"):
            try:
                classes = np.asarray(getattr(clf, "classes_"))
                if classes.size > 0 and np.issubdtype(raw.dtype, np.integer):
                    if raw.min() >= 0 and raw.max() <= (len(classes) - 1):
                        mapped = classes[raw.astype(int)]
                        reason = "clf.classes_ mapping"
                        return np.asarray(mapped), reason
                # if raw values are already in classes, assume they are correct labels
                if set(np.unique(raw)).issubset(set(classes)):
                    mapped = raw
                    reason = "raw_preds_already_in_clf.classes_"
                    return np.asarray(mapped), reason
            except Exception:
                pass

        # 4) If pipeline.report/metrics saved mapping in attr or file, try metrics file path if provided in cfg
        metrics_path = None
        try:
            metrics_path = (self.cfg.get("reports", {}) or {}).get("metrics_path")
        except Exception:
            metrics_path = None

        if metrics_path and os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                if "_label_mapping" in metrics:
                    map_list = np.asarray(metrics["_label_mapping"])
                    if np.issubdtype(raw.dtype, np.integer) and raw.max() <= (len(map_list) - 1):
                        mapped = map_list[raw.astype(int)]
                        reason = f"metrics_file:{metrics_path}"
                        return np.asarray(mapped), reason
            except Exception:
                pass

        # 5) As a last resort, if raw values appear to be original labels, return them
        if mapped is None:
            mapped = raw
            reason = "no_mapping_applied (returned raw_preds)"

        return np.asarray(mapped), reason

    def predict_last(self, x: pd.DataFrame, pipeline: Optional[Any] = None) -> float:
        """
        Predict the last row from the provided batch `x`. If pipeline is provided, use it;
        otherwise load pipeline from configured path.

        Returns the mapped label for the last valid row (original label space if mapping exists).
        """
        # ensure pipeline loaded
        if pipeline is None:
            if self.pipeline is None:
                _ = self.load_pipeline(self.pipeline_path)
            pipeline = self.pipeline
        else:
            # if user provided pipeline object directly, prefer that and also store locally
            self.pipeline = pipeline

        pipeline, clf = self._get_model_and_clf()

        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        # Align expected features (adds missing cols as NaN)
        x_aligned = self._align_columns(x, pipeline)

        # fill using only batch history (ffill / bfill)
        x_aligned = x_aligned.ffill().bfill()

        # drop rows still containing NaNs
        x_valid = x_aligned.dropna(axis=0, how="any")
        if x_valid.empty:
            raise ValueError(
                "All rows contain NaNs after preprocessing/feature engineering. "
                "Increase batch_size in config or ensure streaming rows contain required inputs."
            )

        # take last valid row (most recent in batch)
        last_row = x_valid.tail(1)

        # run prediction. Some pipelines/estimators return 2D arrays; handle that defensively
        try:
            raw_pred = pipeline.predict(last_row)
        except Exception as e:
            # try to call estimator directly if pipeline.predict fails
            # if pipeline is a sklearn Pipeline, call last estimator
            try:
                if clf is not None:
                    raw_pred = clf.predict(last_row)
                else:
                    raise
            except Exception:
                raise RuntimeError(f"pipeline.predict failed and fallback failed: {e}")

        # if raw_pred is 2D (e.g. probabilities or (n,1) shaped), try to reduce to 1D
        raw_pred = np.asarray(raw_pred)
        if raw_pred.ndim > 1:
            # if shape (n,1) -> flatten, if shape (n,k) -> argmax
            if raw_pred.shape[1] == 1:
                raw_pred = raw_pred.ravel()
            else:
                raw_pred = np.argmax(raw_pred, axis=1)

        if raw_pred.size == 0:
            raise RuntimeError("pipeline.predict returned empty result.")

        # map raw preds back to original labels if possible
        mapped_preds, mapping_reason = self._map_predictions_to_original(raw_pred, pipeline, clf)
        # mapped_preds is an array; pick the last element (single row)
        final_val = mapped_preds[-1]

        # try to cast to int if it looks like an integer label, otherwise return as float
        try:
            if np.isfinite(final_val) and float(final_val).is_integer():
                return int(final_val)
        except Exception:
            pass

        # fallback to float
        try:
            return float(final_val)
        except Exception:
            # as ultimate fallback, return as-is
            return final_val

    def run(self, x):
        """
        Backwards-compatible run() method used by your application.
        Equivalent to predict_last().
        """
        return self.predict_last(x)

