# ml/src/pipelines/training.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Any, Dict, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, classification_report, make_scorer

# optional imports guarded at runtime
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    ImbPipeline = Pipeline
    SMOTE = None
    IMBLEARN_AVAILABLE = False

# local helpers (assume exist earlier in file or simple reimplementations)
def compute_metrics_classification(y_true, y_pred):
    """
    Compute a simple dict of classification metrics. Returns a dict ready to be JSON-dumped.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    # return a compact structure
    return {
        "accuracy": float(accuracy),
        "classification_report": report,
    }

def compute_metrics_regression(y_true, y_pred):
    """
    Placeholder regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    return {"mae": float(mae)}

class TrainingPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _prepare_model_and_params(self, model_type: str, base_params: Dict[str, Any]):
        """
        Prepare the model class and model params. Returns (ModelClass, params, is_classifier).
        This is a simplified placeholder - in your real file this will map string to classes.
        """
        # simple mapping for XGBClassifier or other names
        is_classifier = False
        if model_type.lower().startswith("xgb") or "classifier" in model_type.lower():
            is_classifier = True
        # For simplicity, import lazily to avoid heavy imports if not used
        if model_type == "XGBClassifier":
            try:
                from xgboost import XGBClassifier as ModelClass
            except Exception as e:
                raise ImportError("XGBoost not available: " + str(e))
        else:
            # default to sklearn pipeline-friendly estimator - for demo use
            from sklearn.ensemble import RandomForestClassifier as ModelClass
            # Mark classifier if name suggests
            if "classifier" in model_type.lower():
                is_classifier = True

        return ModelClass, base_params, is_classifier

    def run(self, df: pd.DataFrame) -> Pipeline:
        """
        Main training run. Expects df to contain features and target configured in self.cfg.
        Returns the trained pipeline (and saves metrics + artifact).
        """
        # --- read training config ---
        training_cfg = self.cfg.get("training", {})
        target_cfg = training_cfg.get("target", {}) or {}
        source_col = target_cfg.get("source_col", "quality")
        target_name = target_cfg.get("target_name", "target")
        # If horizon or other transforms required, that would be applied earlier; assume target column exists
        if source_col not in df.columns:
            raise ValueError(f"Target column '{source_col}' not in dataframe")

        # create features and target
        feature_cols = [c for c in df.columns if c not in [target_name, source_col]]
        X = df[feature_cols].copy()
        y = df[source_col].copy()

        if X.shape[1] == 0:
            raise ValueError("No features selected after dropping target/source/datetime-like columns.")

        frac = float(self.cfg["training"].get("train_fraction", 0.8))
        model_type = self.cfg["training"].get("model_type", "XGBClassifier")
        base_params = self.cfg["training"].get("model_params", {}) or {}

        # Prepare model class / params / classifier flag
        ModelClass, model_params, is_classifier = self._prepare_model_and_params(model_type, base_params)

        # If classifier: factorize labels to 0..K-1 and keep mapping for reporting
        label_mapping = None
        if is_classifier:
            # pd.factorize returns (labels_as_ints, unique_values_index)
            y, label_mapping = pd.factorize(y)  # y now ints 0..K-1

        # Train/test split (stratify for classification)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1.0 - frac), shuffle=True, stratify=(y if is_classifier else None), random_state=42
        )

        # Build pipeline: include SMOTE optionally for classification
        use_smote = bool(self.cfg.get("training", {}).get("use_smote", False)) and IMBLEARN_AVAILABLE and is_classifier

        if use_smote:
            pipeline = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)), ("clf", ModelClass(**model_params))])
        else:
            pipeline = Pipeline([("scaler", StandardScaler()), ("clf", ModelClass(**model_params))])

        # Fit
        pipeline.fit(X_train, y_train)

        # Predict & metrics
        y_pred = pipeline.predict(X_test)

        metrics = {}
        if is_classifier:
            # If predict returns probabilities, convert to labels (rare for sklearn predict)
            try:
                if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 and y_pred.shape[1] > 1:
                    y_pred = np.asarray(y_pred).argmax(axis=1)
            except Exception:
                pass

            metrics = compute_metrics_classification(y_test, y_pred)

            # Convert classification_report label keys back to original labels for readability
            if "classification_report" in metrics and isinstance(metrics["classification_report"], dict) and label_mapping is not None:
                cr = metrics["classification_report"]
                remapped = {}
                for k, v in cr.items():
                    # numeric string keys correspond to labels; keep aggregates as-is
                    try:
                        ik = int(k)
                        orig_label = label_mapping[ik]
                        remapped[str(int(orig_label))] = v
                    except Exception:
                        remapped[k] = v
                metrics["classification_report"] = remapped

            # also save the mapping for future reference
            if label_mapping is not None:
                # ensure mapping is serializable: cast numeric labels to int where possible
                def _safe_repr(x):
                    try:
                        if isinstance(x, (np.integer, int)) or (isinstance(x, float) and not np.isnan(x) and float(x).is_integer()):
                            return int(x)
                        return str(x)
                    except Exception:
                        return str(x)
                metrics["_label_mapping"] = [_safe_repr(x) for x in label_mapping]

        else:
            metrics = compute_metrics_regression(y_test, y_pred)

        # Save metrics
        metrics_path = self.cfg.get("reports", {}).get("metrics_path", "reports/metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # attach mapping to pipeline so consumers can inverse-map predictions
        if label_mapping is not None:
            # make mapping explicit and indexable (numpy array)
            pipeline.label_mapping_ = np.asarray(label_mapping)

        # save trained pipeline artifact so attached attrs persist
        model_path = self.cfg.get('artifacts', {}).get('model_path', 'models/pipeline.joblib')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            joblib.dump(pipeline, model_path)
        except Exception as e:
            print(f"Warning: failed to save pipeline to {model_path}: {e}")

        # Return pipeline (with label_mapping_ attached and artifact saved)
        return pipeline
