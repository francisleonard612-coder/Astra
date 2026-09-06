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
    name = "xgboost"

    def __init__(self, buffer_size: int = 5000):
        super().__init__(buffer_size)
        try:
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                objective="multi:softprob", num_class=N_DIGITS,
                eval_metric="mlogloss", n_jobs=-1, verbosity=0,
            )
            self._available = True
        except ImportError:
            self._available = False
            self._model = None

    def is_ready(self, state: SymbolState) -> bool:
        return self._available and self._fitted

    def retrain(self) -> None:
        if not self._available:
            return
        super().retrain()
