"""
Model definitions and training.

Each model is wrapped in a scikit-learn Pipeline with a StandardScaler, so the
scaling statistics are fitted on the training sample only (no leakage).
Scaling does not change tree-based models but keeps the pipeline uniform.
"""

import warnings

from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
        "NN 1 layer": MLPClassifier(hidden_layer_sizes=config.MLP_SHALLOW_LAYERS,
                                    random_state=seed, **config.MLP_COMMON),
        "NN 3 layers": MLPClassifier(hidden_layer_sizes=config.MLP_DEEP_LAYERS,
                                     random_state=seed, **config.MLP_COMMON),
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
    for name, model in models.items():
        # On pure noise the networks may hit max_iter without converging; that
        # is expected here, so print one short note instead of a long warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            print(f"  note: {name} stopped at max_iter without converging")
    return models
