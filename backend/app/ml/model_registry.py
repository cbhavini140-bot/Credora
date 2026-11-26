"""Registry and inference/explainability wrapper for the three CREDORA .joblib models."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import core as xgb_core

from app.core.config import get_settings
from app.ml.feature_engineering import (
    PAYSIM_MODEL_FEATURES,
    SPARKOV_MODEL_FEATURES,
    ULB_FEATURES,
    prepare_features_for_model,
)


def _parse_ubjson(buf: bytes | bytearray) -> Any:
    """Minimal UBJSON decoder to recover XGBoost Model JSON when Booster.__setstate__ hits C++ stream checks."""
    pos = [0]

    def read(n: int) -> bytes:
        s = buf[pos[0] : pos[0] + n]
        pos[0] += n
        return bytes(s)

    def read_int(marker: bytes) -> int:
        if marker == b"i":
            return struct.unpack(">b", read(1))[0]
        if marker == b"U":
            return struct.unpack(">B", read(1))[0]
        if marker == b"I":
            return struct.unpack(">h", read(2))[0]
        if marker == b"l":
            return struct.unpack(">i", read(4))[0]
        if marker == b"L":
            return struct.unpack(">q", read(8))[0]
        raise ValueError(f"Unsupported UBJSON int marker {marker!r} at offset {pos[0]}")

    def read_str() -> str:
        length = read_int(read(1))
        return read(length).decode("utf-8", errors="replace")

    def parse_val(marker: Optional[bytes] = None) -> Any:
        if marker is None:
            marker = read(1)
        if marker == b"Z":
            return None
        if marker == b"T":
            return True
        if marker == b"F":
            return False
        if marker in (b"i", b"U", b"I", b"l", b"L"):
            return read_int(marker)
        if marker == b"d":
            return struct.unpack(">f", read(4))[0]
        if marker == b"D":
            return struct.unpack(">d", read(8))[0]
        if marker == b"S":
            return read_str()
        if marker == b"[":
            arr: list[Any] = []
            nxt = read(1)
            if nxt == b"$":
                el_marker = read(1)
                count_marker = read(1)
                if count_marker != b"#":
                    raise ValueError(f"Expected '#' in typed UBJSON array, got {count_marker!r}")
                count = read_int(read(1))
                for _ in range(count):
                    arr.append(parse_val(el_marker))
                return arr
            if nxt == b"#":
                count = read_int(read(1))
                for _ in range(count):
                    arr.append(parse_val())
                return arr
            while nxt != b"]":
                arr.append(parse_val(nxt))
                nxt = read(1)
            return arr
        if marker == b"{":
            obj: dict[str, Any] = {}
            nxt = read(1)
            while nxt != b"}":
                pos[0] -= 1
                key = read_str()
                obj[key] = parse_val()
                nxt = read(1)
            return obj
        raise ValueError(f"Unknown UBJSON marker {marker!r} at offset {pos[0]}")

    return parse_val()


def load_joblib_package(filepath: Path) -> dict[str, Any]:
    """Load a CREDORA .joblib artifact, handling cross-OS XGBoost Booster UBJSON compatibility."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Required ML artifact is missing: {filepath}. "
            "Place the trained .joblib file in credora/ml_models/ before starting inference."
        )

    orig_setstate = xgb_core.Booster.__setstate__

    def compat_setstate(self, state: dict) -> None:
        try:
            orig_setstate(self, state)
        except Exception:
            raw_handle = state.get("handle")
            if not raw_handle:
                raise
            parsed = _parse_ubjson(raw_handle)
            model_obj = parsed.get("Model")
            if not model_obj:
                raise
            dmats = xgb_core.c_array(xgb_core.ctypes.c_void_p, [])
            handle = xgb_core.ctypes.c_void_p()
            xgb_core._check_call(
                xgb_core._LIB.XGBoosterCreate(dmats, xgb_core.c_bst_ulong(0), xgb_core.ctypes.byref(handle))
            )
            self.handle = handle
            self.load_model(bytearray(json.dumps(model_obj).encode("utf-8")))

    xgb_core.Booster.__setstate__ = compat_setstate
    try:
        pkg = joblib.load(filepath)
    finally:
        xgb_core.Booster.__setstate__ = orig_setstate

    if not isinstance(pkg, dict) or "model" not in pkg or "preprocessor" not in pkg:
        raise ValueError(f"Invalid CREDORA model package format in {filepath.name}.")
    return pkg


def _clean_feature_name(raw_name: str, model_key: str) -> str:
    """Format ColumnTransformer output feature names into clean analyst-readable labels."""
    name = raw_name
    for prefix in ("scaled_features__", "remainder__", "numeric__", "categorical__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if model_key == "ulb" and name.startswith("V") and name[1:].isdigit():
        return f"{name} (PCA Component)"
    return name


def _base_feature_name(raw_name: str, model_key: str) -> str:
    cleaned = raw_name
    for prefix in ("scaled_features__", "remainder__", "numeric__", "categorical__"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if model_key == "sparkov":
        for cat in ("category_", "gender_", "state_"):
            if cleaned.startswith(cat):
                return cat[:-1]
    if model_key == "paysim" and cleaned.startswith("type_"):
        return "type"
    return cleaned


@dataclass
class LoadedModelBundle:
    model_key: str
    artifact_filename: str
    model_name: str
    dataset_name: str
    algorithm: str
    threshold: float
    expected_features: list[str]
    preprocessor: Any
    model: Any
    output_feature_names: list[str]
    display_feature_names: list[str]
    explainer: Any
    explainability_note: str


class CredoraModelRegistry:
    """Holds all three independent CREDORA models loaded from .joblib artifacts."""

    ARTIFACT_SPECS = {
        "ulb": {
            "filename": "credora_ulb_model.joblib",
            "algorithm": "RandomForestClassifier",
            "features": ULB_FEATURES,
            "explainability_note": (
                "Features V1-V28 are anonymized PCA components from the ULB European credit card dataset, "
                "not human-interpretable business concepts. Time and Amount are standard-scaled."
            ),
        },
        "sparkov": {
            "filename": "credora_sparkov_model.joblib",
            "algorithm": "XGBClassifier",
            "features": SPARKOV_MODEL_FEATURES,
            "explainability_note": (
                "Explanations use actual engineered Sparkov features including transaction amount, "
                "city population, temporal components, Haversine merchant distance (km), category, gender, and state."
            ),
        },
        "paysim": {
            "filename": "credora_paysim_model.joblib",
            "algorithm": "XGBClassifier",
            "features": PAYSIM_MODEL_FEATURES,
            "explainability_note": (
                "Explanations use actual PaySim synthetic transaction features including balance changes, "
                "balance reconciliation errors (origin_error, destination_error), hour of day, and transaction type."
            ),
        },
    }

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = models_dir or get_settings().ml_models_path
        self.bundles: dict[str, LoadedModelBundle] = {}
        self.load_errors: dict[str, str] = {}

    def load_all(self) -> None:
        """Load each .joblib package independently without substituting models on failure."""
        self.bundles.clear()
        self.load_errors.clear()

        for key, spec in self.ARTIFACT_SPECS.items():
            fpath = self.models_dir / spec["filename"]
            try:
                pkg = load_joblib_package(fpath)
                preprocessor = pkg["preprocessor"]
                model = pkg["model"]

                try:
                    raw_out = list(preprocessor.get_feature_names_out())
                except Exception:
                    raw_out = list(spec["features"])

                display_names = [_clean_feature_name(n, key) for n in raw_out]
                explainer = shap.TreeExplainer(model)

                self.bundles[key] = LoadedModelBundle(
                    model_key=key,
                    artifact_filename=spec["filename"],
                    model_name=str(pkg.get("model_name", f"CREDORA {key.upper()}")),
                    dataset_name=str(pkg.get("dataset", key.upper())),
                    algorithm=spec["algorithm"],
                    threshold=float(pkg.get("threshold", 0.5)),
                    expected_features=list(spec["features"]),
                    preprocessor=preprocessor,
                    model=model,
                    output_feature_names=raw_out,
                    display_feature_names=display_names,
                    explainer=explainer,
                    explainability_note=spec["explainability_note"],
                )
            except Exception as exc:
                self.load_errors[key] = str(exc)

    def get_bundle(self, model_key: str) -> LoadedModelBundle:
        key = model_key.lower().strip()
        if key in self.bundles:
            return self.bundles[key]
        if key in self.load_errors:
            raise RuntimeError(f"Model '{key}' failed to load: {self.load_errors[key]}")
        raise ValueError(f"Unknown model key '{model_key}'. Valid options: ulb, sparkov, paysim.")

    def status_summary(self) -> dict[str, Any]:
        return {
            key: {
                "loaded": key in self.bundles,
                "error": self.load_errors.get(key),
                "artifact_filename": spec["filename"],
                "model_name": self.bundles[key].model_name if key in self.bundles else None,
                "dataset_name": self.bundles[key].dataset_name if key in self.bundles else None,
                "threshold": self.bundles[key].threshold if key in self.bundles else None,
            }
            for key, spec in self.ARTIFACT_SPECS.items()
        }

    @staticmethod
    def compute_risk_level(
        risk_score: float, medium_threshold: float = 30.0, high_threshold: float = 70.0
    ) -> str:
        if risk_score >= high_threshold:
            return "HIGH"
        if risk_score >= medium_threshold:
            return "MEDIUM"
        return "LOW"

    def predict_dataframe(
        self,
        model_key: str,
        raw_df: pd.DataFrame,
        medium_threshold: float = 30.0,
        high_threshold: float = 70.0,
        include_shap: bool = False,
    ) -> tuple[list[dict[str, Any]], pd.DataFrame, np.ndarray]:
        """Run preprocessing, model probability prediction, risk scoring, and optional SHAP."""
        bundle = self.get_bundle(model_key)
        prepared_df = prepare_features_for_model(bundle.model_key, raw_df)
        X_transformed = bundle.preprocessor.transform(prepared_df)

        probs = bundle.model.predict_proba(X_transformed)[:, 1]
        results: list[dict[str, Any]] = []

        shap_matrix = None
        if include_shap:
            shap_matrix = self._compute_shap_matrix(bundle, X_transformed)

        for idx, prob in enumerate(probs):
            prob_f = float(prob)
            risk_score = round(prob_f * 100.0, 2)
            risk_level = self.compute_risk_level(risk_score, medium_threshold, high_threshold)
            predicted_fraud = bool(prob_f >= bundle.threshold)
            item: dict[str, Any] = {
                "fraud_probability": prob_f,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "predicted_fraud": predicted_fraud,
                "threshold_used": bundle.threshold,
            }
            if shap_matrix is not None:
                item["shap_explanation"] = self._format_single_shap(
                    bundle, shap_matrix[idx], X_transformed[idx]
                )
            results.append(item)

        return results, prepared_df, X_transformed

    def _compute_shap_matrix(self, bundle: LoadedModelBundle, X_transformed: Any) -> np.ndarray:
        dense_X = X_transformed.toarray() if hasattr(X_transformed, "toarray") else np.asarray(X_transformed)
        sv = bundle.explainer.shap_values(dense_X)
        if isinstance(sv, list):
            # Older SHAP returns [class_0_array, class_1_array]
            arr = np.asarray(sv[1] if len(sv) > 1 else sv[0])
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3 and arr.shape[-1] == 2:
                # Newer SHAP for binary RandomForest returns (n_samples, n_features, 2)
                arr = arr[:, :, 1]
        return arr

    def _format_single_shap(
        self, bundle: LoadedModelBundle, row_shap: np.ndarray, row_x: Any, top_k: int = 10
    ) -> dict[str, Any]:
        vals = row_x.toarray().ravel() if hasattr(row_x, "toarray") else np.asarray(row_x).ravel()
        contributions = []
        for i, shap_val in enumerate(row_shap):
            fname = (
                bundle.display_feature_names[i]
                if i < len(bundle.display_feature_names)
                else f"feature_{i}"
            )
            raw_fname = (
                bundle.output_feature_names[i]
                if i < len(bundle.output_feature_names)
                else fname
            )
            contributions.append(
                {
                    "feature": fname,
                    "raw_feature": raw_fname,
                    "shap_value": round(float(shap_val), 6),
                    "abs_shap": round(abs(float(shap_val)), 6),
                    "transformed_value": round(float(vals[i]), 6) if i < len(vals) else None,
                    "direction": "increases_risk" if float(shap_val) >= 0 else "decreases_risk",
                }
            )
        contributions.sort(key=lambda c: c["abs_shap"], reverse=True)
        expected_val = bundle.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_val = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])
        else:
            base_val = float(expected_val)

        return {
            "model_key": bundle.model_key,
            "model_name": bundle.model_name,
            "base_value": round(base_val, 6),
            "explainability_note": bundle.explainability_note,
            "top_contributions": contributions[:top_k],
            "positive_contributors": [c for c in contributions if c["shap_value"] > 0][:5],
            "negative_contributors": [c for c in contributions if c["shap_value"] < 0][:5],
        }

    def explain_single_row(self, model_key: str, raw_row: dict[str, Any]) -> dict[str, Any]:
        """Compute SHAP explanation for a single stored or submitted transaction dictionary."""
        bundle = self.get_bundle(model_key)
        df = pd.DataFrame([raw_row])
        preds, _, _ = self.predict_dataframe(model_key, df, include_shap=True)
        result = preds[0]
        result["explainability_note"] = bundle.explainability_note
        return result

    def global_feature_importance(self, model_key: str, top_k: int = 15) -> dict[str, Any]:
        """Return genuine model feature importances mapped to real feature names."""
        bundle = self.get_bundle(model_key)
        importances = getattr(bundle.model, "feature_importances_", None)
        if importances is None:
            raise ValueError(f"Model '{model_key}' does not expose feature_importances_.")

        detailed = []
        grouped: dict[str, float] = {}
        for i, imp in enumerate(importances):
            raw_name = (
                bundle.output_feature_names[i]
                if i < len(bundle.output_feature_names)
                else f"feature_{i}"
            )
            display_name = (
                bundle.display_feature_names[i]
                if i < len(bundle.display_feature_names)
                else raw_name
            )
            imp_f = float(imp)
            detailed.append(
                {
                    "feature": display_name,
                    "raw_feature": raw_name,
                    "importance": round(imp_f, 6),
                }
            )
            base_name = _base_feature_name(raw_name, bundle.model_key)
            grouped[base_name] = grouped.get(base_name, 0.0) + imp_f

        detailed.sort(key=lambda x: x["importance"], reverse=True)
        grouped_list = [
            {"feature": k, "importance": round(v, 6)}
            for k, v in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
        ]

        return {
            "model_key": bundle.model_key,
            "model_name": bundle.model_name,
            "dataset_name": bundle.dataset_name,
            "algorithm": bundle.algorithm,
            "explainability_note": bundle.explainability_note,
            "transformed_feature_importance": detailed[:top_k],
            "grouped_feature_importance": grouped_list[:top_k],
        }


_registry_instance: Optional[CredoraModelRegistry] = None


def get_model_registry() -> CredoraModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = CredoraModelRegistry()
        _registry_instance.load_all()
    return _registry_instance
