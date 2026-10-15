"""
Model definitions and training.

Each model is wrapped in a scikit-learn Pipeline with a StandardScaler, so the
scaling statistics are fitted on the training sample only (no leakage).
Scaling does not change tree-based models but keeps the pipeline uniform.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config


def get_models(seed):
    """Return a dict of un-fitted models with the hyperparameters in config."""
    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(**config.LOGREG_PARAMS),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, **config.TREE_PARAMS),
        "Random Forest": RandomForestClassifier(random_state=seed, **config.FOREST_PARAMS),
    }
    if config.USE_XGBOOST:
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(random_state=seed, **config.XGB_PARAMS)
        except ImportError:
            print("  xgboost not installed -> skipping XGBoost")

    return {name: Pipeline([("scaler", StandardScaler()), ("model", m)])
            for name, m in models.items()}


def train_models(models, X_train, y_train):
    """Fit every model on the training sample only."""
    for model in models.values():
        model.fit(X_train, y_train)
    return models
