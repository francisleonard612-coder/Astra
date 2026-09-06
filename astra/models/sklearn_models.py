from __future__ import annotations

import numpy as np

from features.feature_engine import FeatureBundle
from models.base import DigitModel, N_DIGITS, normalize
from state.rolling_state import SymbolState


class LogisticModel(DigitModel):
    """Online multinomial logistic regression via SGDClassifier(loss='log_loss').
    Unlike sklearn's plain LogisticRegression, SGDClassifier supports
    partial_fit, so this one genuinely learns incrementally, tick by tick,
    rather than needing scheduled batch retraining."""
    name = "logistic"

    def __init__(self):
        from sklearn.linear_model import SGDClassifier
        self._clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1, warm_start=True)
        self._fitted = False
        self._classes = np.arange(N_DIGITS)

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        if not self._fitted or bundle.vector is None:
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        try:
            proba = self._clf.predict_proba(bundle.vector.reshape(1, -1))[0]
        except Exception:  # noqa: BLE001
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        full = np.full(N_DIGITS, 1e-6)
        for cls, p in zip(self._clf.classes_, proba):
            full[int(cls)] = p
        return normalize(full)

    def observe(self, state: SymbolState, bundle: FeatureBundle, actual_digit: int) -> None:
        if bundle.vector is None:
            return
        x = bundle.vector.reshape(1, -1)
        y = np.array([actual_digit])
        if not self._fitted:
            self._clf.partial_fit(x, y, classes=self._classes)
            self._fitted = True
        else:
            self._clf.partial_fit(x, y)

    def is_ready(self, state: SymbolState) -> bool:
        return self._fitted and state.total_observed >= 100


class _BatchModel(DigitModel):
    """Shared scaffolding for models that only support batch (not incremental)
    fitting -- Random Forest and XGBoost. `observe` just buffers samples;
    the actual `.fit()` call happens in learning/retraining.py on a cadence,
    matching spec section 11 ("do not retrain heavyweight models after every
    tick")."""

    def __init__(self, buffer_size: int = 5000):
        self._buffer_size = buffer_size
        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._model = None
        self._fitted = False

    def observe(self, state: SymbolState, bundle: FeatureBundle, actual_digit: int) -> None:
        if bundle.vector is None:
            return
        self._X.append(bundle.vector)
        self._y.append(actual_digit)
        if len(self._X) > self._buffer_size:
            self._X.pop(0)
            self._y.pop(0)

    def ready_to_retrain(self, min_observations: int) -> bool:
        return len(self._X) >= min_observations

    def retrain(self) -> None:
        if len(self._X) < 30:
            return
        X = np.vstack(self._X)
        y = np.array(self._y)
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        if not self._fitted or bundle.vector is None:
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        try:
            proba = self._model.predict_proba(bundle.vector.reshape(1, -1))[0]
        except Exception:  # noqa: BLE001
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        full = np.full(N_DIGITS, 1e-6)
        classes = getattr(self._model, "classes_", np.arange(len(proba)))
        for cls, p in zip(classes, proba):
            full[int(cls)] = p
        return normalize(full)

    def is_ready(self, state: SymbolState) -> bool:
        return self._fitted


class RandomForestModel(_BatchModel):
    name = "random_forest"

    def __init__(self, buffer_size: int = 5000):
        super().__init__(buffer_size)
        from sklearn.ensemble import RandomForestClassifier
        self._model = RandomForestClassifier(n_estimators=150, max_depth=8, n_jobs=-1, random_state=42)


class XGBoostModel(_BatchModel):
    """Uses XGBoost's low-level Learning API (Booster + DMatrix) instead of
    the XGBClassifier sklearn wrapper.

    Root cause of the production failure ("Batch model retrain failed" /
    ValueError: Invalid classes inferred from unique values of `y`", every
    retrain cycle, xgboost permanently stuck unfitted): this is NOT an
    instance-reuse/warm-start issue -- a brand-new, never-before-fitted
    XGBClassifier raises the exact same error. XGBoost's sklearn wrapper
    (confirmed against xgboost 3.4.1) requires the labels passed to .fit()
    to be a contiguous integer range starting at 0 (0..8, or 0..9, etc) --
    it does NOT accept a set like {0,1,2,3,4,5,6,8,9} that skips a value in
    the middle (missing "7") even though every value is a valid digit and
    num_class=10 was declared explicitly. Since `_X`/`_y` here is a sliding
    window (see observe()), any retrain whose window happens not to contain
    one particular digit -- entirely normal for last-digit-of-price data --
    trips this check, and once it does, it does so on essentially every
    subsequent retrain too (only a window containing all 10 digits, or one
    missing only the top digit 9, passes).

    The low-level Booster API has no such restriction: `label` is used
    directly as a class index into the `num_class`-sized output declared in
    `params`, so a window missing one or more digits just means those
    classes get no training signal in that round -- no validation error --
    and prediction always returns a full-length num_class probability
    vector regardless of which digits happened to appear in the last
    training window.
    """
    name = "xgboost"

    def __init__(self, buffer_size: int = 5000):
        super().__init__(buffer_size)
        try:
            import xgboost  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        self._booster = None
        self._params = {
            "objective": "multi:softprob", "num_class": N_DIGITS,
            "max_depth": 5, "eta": 0.05, "eval_metric": "mlogloss",
            "nthread": -1, "verbosity": 0,
        }
        self._num_boost_round = 200

    def is_ready(self, state: SymbolState) -> bool:
        return self._available and self._fitted

    def retrain(self) -> None:
        if not self._available or len(self._X) < 30:
            return
        import xgboost as xgb
        X = np.vstack(self._X)
        y = np.array(self._y)
        dtrain = xgb.DMatrix(X, label=y)
        self._booster = xgb.train(self._params, dtrain, num_boost_round=self._num_boost_round)
        self._fitted = True

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        if not self._fitted or bundle.vector is None:
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        try:
            import xgboost as xgb
            proba = self._booster.predict(xgb.DMatrix(bundle.vector.reshape(1, -1)))[0]
        except Exception:  # noqa: BLE001
            return np.full(N_DIGITS, 1.0 / N_DIGITS)
        return normalize(proba)
