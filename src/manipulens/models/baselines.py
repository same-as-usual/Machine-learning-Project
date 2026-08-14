"""Baseline model ladder for clickbait intensity.

Rung 1: lexicon rules only (labeling-function mean) — no learning.
Rung 2: TF-IDF (word + char n-grams) + Logistic Regression.
Rung 3: LightGBM on engineered LF/structural features + TF-IDF LogReg probability.

All rungs stay in the eval table forever — the transformer (Phase 3) must beat
them to be promoted. The winning rung is calibrated on val and saved as the
serving artifact.

Usage: python -m manipulens.models.baselines
Writes: models/artifacts/baseline.joblib, reports/eval_report.json
"""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from manipulens.config import REPO_ROOT, artifacts_dir, data_dir, load_params
from manipulens.eval.metrics import classification_report_dict
from manipulens.labeling.labeling_functions import FEATURE_NAMES, feature_vector
from manipulens.models.calibrate import Calibrator


def lf_features(headlines: pd.Series) -> np.ndarray:
    rows = [feature_vector(h) for h in headlines]
    return np.array([[r[name] for name in FEATURE_NAMES] for r in rows])


class BaselinePipeline:
    """TF-IDF + LogReg and LightGBM-on-features, with a shared calibrator."""

    def __init__(self, params: dict):
        tfidf = params["tfidf"]
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, tfidf["word_ngram_max"]),
            max_features=tfidf["max_features"],
            lowercase=True,
            strip_accents="unicode",
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(tfidf["char_ngram_min"], tfidf["char_ngram_max"]),
            max_features=tfidf["max_features"],
            lowercase=True,
        )
        self.logreg = LogisticRegression(max_iter=2000, C=1.0)
        lgbm = params["lightgbm"]
        self.gbm = LGBMClassifier(
            n_estimators=lgbm["n_estimators"],
            learning_rate=lgbm["learning_rate"],
            num_leaves=lgbm["num_leaves"],
            random_state=params["seed"],
            verbose=-1,
        )
        self.calibrator: Calibrator | None = None
        self.calibration_method = params["calibration"]

    # --- representation ---
    def _text_matrix(self, headlines: pd.Series, fit: bool = False):
        if fit:
            w = self.word_vec.fit_transform(headlines)
            c = self.char_vec.fit_transform(headlines)
        else:
            w = self.word_vec.transform(headlines)
            c = self.char_vec.transform(headlines)
        return hstack([w, c]).tocsr()

    # --- training ---
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, dict[str, float]]:
        x_train_text = self._text_matrix(train["headline"], fit=True)
        x_val_text = self._text_matrix(val["headline"])
        y_train, y_val = train["label_clickbait"].to_numpy(), val["label_clickbait"].to_numpy()

        self.logreg.fit(x_train_text, y_train)
        p_train_lr = self.logreg.predict_proba(x_train_text)[:, 1]
        p_val_lr = self.logreg.predict_proba(x_val_text)[:, 1]

        f_train = np.column_stack([lf_features(train["headline"]), p_train_lr])
        f_val = np.column_stack([lf_features(val["headline"]), p_val_lr])
        self.gbm.fit(f_train, y_train)
        p_val_gbm = self.gbm.predict_proba(f_val)[:, 1]

        self.calibrator = Calibrator(self.calibration_method).fit(p_val_gbm, y_val)
        return {
            "val_logreg": classification_report_dict(y_val, p_val_lr),
            "val_lgbm_raw": classification_report_dict(y_val, p_val_gbm),
            "val_lgbm_calibrated": classification_report_dict(y_val, self.calibrator.transform(p_val_gbm)),
        }

    # --- inference ---
    def predict_proba(self, headlines: pd.Series | list[str], calibrated: bool = True) -> np.ndarray:
        headlines = pd.Series(headlines)
        x_text = self._text_matrix(headlines)
        p_lr = self.logreg.predict_proba(x_text)[:, 1]
        feats = np.column_stack([lf_features(headlines), p_lr])
        p = self.gbm.predict_proba(feats)[:, 1]
        if calibrated and self.calibrator is not None:
            p = self.calibrator.transform(p)
        return p


def rule_baseline_proba(headlines: pd.Series) -> np.ndarray:
    """Rung 1: mean of labeling-function scores. No learning at all."""
    feats = lf_features(headlines)
    lf_cols = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("lf_")]
    return feats[:, lf_cols].mean(axis=1).clip(0, 1)


def main(argv: list[str] | None = None) -> None:
    params = load_params()["train"]
    processed = data_dir("processed")
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    y_test = test["label_clickbait"].to_numpy()

    report: dict = {"n_train": len(train), "n_val": len(val), "n_test": len(test)}

    # Rung 1: rules only
    report["test_rules_only"] = classification_report_dict(y_test, rule_baseline_proba(test["headline"]))

    # Rungs 2+3
    pipe = BaselinePipeline(params)
    report.update(pipe.fit(train, val))
    p_test_raw = pipe.predict_proba(test["headline"], calibrated=False)
    p_test_cal = pipe.predict_proba(test["headline"], calibrated=True)
    report["test_lgbm_raw"] = classification_report_dict(y_test, p_test_raw)
    report["test_lgbm_calibrated"] = classification_report_dict(y_test, p_test_cal)
    report["calibration_method"] = pipe.calibrator.method if pipe.calibrator else "none"

    model_path = artifacts_dir("models") / "baseline.joblib"
    joblib.dump(pipe, model_path)
    report_path = REPO_ROOT / load_params()["artifacts"]["reports_dir"] / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nmodel -> {model_path}\nreport -> {report_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
