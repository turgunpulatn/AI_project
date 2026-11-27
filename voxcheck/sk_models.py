"""Small sklearn helpers that must be importable when a saved model is loaded."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class FastSVM(BaseEstimator, ClassifierMixin):
    """RBF-SVM whose decision value is squashed with a sigmoid.

    SVC(probability=True) runs an extra internal 5-fold CV (Platt scaling) -> ~5x slower.
    A sigmoid of the margin is monotonic, so AUC / EER are identical."""

    def __init__(self, C=3.0, max_train=8000):
        self.C, self.max_train = C, max_train

    def fit(self, X, y):
        if len(y) > self.max_train:  # SVM is O(n^2): subsample very large sets
            idx = np.random.default_rng(0).choice(len(y), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.pipe_ = make_pipeline(StandardScaler(), SVC(C=self.C, class_weight="balanced", cache_size=1000))
        self.pipe_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-2 * self.pipe_.decision_function(X)))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
