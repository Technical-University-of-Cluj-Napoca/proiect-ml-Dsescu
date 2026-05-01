"""
Utilitare comune pentru proiectul ML:
- EDA
- preprocesare
- antrenare modele baseline
- ajustare hiperparametri pentru toti algoritmii
- curbe de invatare
- SHAP global si local
- salvare artefacte pentru Streamlit

Date folosite:
- clasificare: data/heart.csv, target = HeartDisease
- regresie: data/insurance.csv, target = charges
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
except Exception:
    ExplainableBoostingClassifier = None
    ExplainableBoostingRegressor = None

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real
except Exception:  
    BayesSearchCV = None
    Categorical = Integer = Real = None

try:
    import shap
except Exception:  
    shap = None

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CLASSIFICATION_TARGET = "HeartDisease"
REGRESSION_TARGET = "charges"

def safe_name(name: str) -> str:
    """Transforma numele modelului intr-un nume potrivit pentru fisiere."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Incarca un fisier CSV si intoarce DataFrame-ul."""
    return pd.read_csv(csv_path)


def get_feature_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa variabilele de intrare X de variabila tinta y."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifica automat coloanele numerice si categorice."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibil cu versiuni diferite de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Creeaza preprocesorul:
    - numeric: imputare cu mediana + standardizare
    - categoric: imputare cu moda + one-hot encoding
    Output-ul este dens, util pentru GaussianNB, SVM, SHAP etc.
    """
    numeric_cols, categorical_cols = infer_columns(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    target: str,
    task: str,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Imparte datele in 75% train si 25% test/validare."""
    X, y = get_feature_target(df, target)
    stratify = y if task == "classification" else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def make_pipeline(X: pd.DataFrame, model: Any) -> Pipeline:
    """Creeaza pipeline complet: preprocesare + model."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


def get_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    """Extrage numele coloanelor dupa one-hot encoding."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transformers_.shape[0])]


def transform_with_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Aplica doar preprocesorul din pipeline."""
    Xt = pipeline.named_steps["preprocessor"].transform(X)
    Xt = np.asarray(Xt)
    feature_names = get_feature_names_from_pipeline(pipeline)
    return Xt, feature_names


def _require_optional(package_name: str, estimator: Any) -> None:
    if estimator is None:
        raise ImportError(
            f"Pachetul pentru {package_name} nu este instalat. "
            f"Instaleaza dependintele din requirements.txt."
        )


def get_classification_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Toti algoritmii ceruti pentru clasificare."""
    _require_optional("XGBoost", XGBClassifier)
    _require_optional("CatBoost", CatBoostClassifier)
    _require_optional("Explainable Boosting Machine", ExplainableBoostingClassifier)

    return {
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "Support Vector Machine": SVC(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
        "Explainable Boosting Machine": ExplainableBoostingClassifier(random_state=random_state),
    }


def get_regression_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Toti algoritmii ceruti pentru regresie."""
    _require_optional("XGBoost", XGBRegressor)
    _require_optional("CatBoost", CatBoostRegressor)
    _require_optional("Explainable Boosting Machine", ExplainableBoostingRegressor)

    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "Support Vector Regressor": SVR(),
        "K-Nearest Neighbor Regressor": KNeighborsRegressor(),
        "Gaussian Process Regressor": GaussianProcessRegressor(random_state=random_state, normalize_y=True),
        "XGBoost Regressor": XGBRegressor(
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        ),
        "CatBoost Regressor": CatBoostRegressor(
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
        "Explainable Boosting Regressor": ExplainableBoostingRegressor(random_state=random_state),
    }


def get_models(task: str, random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    if task == "classification":
        return get_classification_models(random_state)
    if task == "regression":
        return get_regression_models(random_state)
    raise ValueError("task trebuie sa fie 'classification' sau 'regression'")


def get_classification_scores(estimator: Pipeline, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    """Scor continuu pentru ROC-AUC: probabilitate clasa 1 sau decision_function."""
    try:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
    except Exception:
        pass

    try:
        if hasattr(estimator, "decision_function"):
            return estimator.decision_function(X_test)
    except Exception:
        pass

    return None


def evaluate_classification(estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = estimator.predict(X_test)
    y_score = get_classification_scores(estimator, X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_test, y_score)
        except Exception:
            metrics["ROC_AUC"] = np.nan
    else:
        metrics["ROC_AUC"] = np.nan

    cm = confusion_matrix(y_test, y_pred)
    metrics["TN"] = int(cm[0, 0])
    metrics["FP"] = int(cm[0, 1])
    metrics["FN"] = int(cm[1, 0])
    metrics["TP"] = int(cm[1, 1])
    return metrics


def evaluate_regression(estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    return {
        "MSE": mse,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_test, y_pred),
    }


def evaluate_model(task: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    if task == "classification":
        return evaluate_classification(estimator, X_test, y_test)
    return evaluate_regression(estimator, X_test, y_test)


def rank_results(task: str, results_df: pd.DataFrame) -> pd.DataFrame:
    """Ordoneaza modelele in functie de metricile relevante."""
    df = results_df.copy()
    if task == "classification":
        sort_cols = [c for c in ["F1", "ROC_AUC", "Accuracy"] if c in df.columns]
        return df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    sort_cols = [c for c in ["RMSE", "MAE", "R2"] if c in df.columns]
    ascending = [True, True, False][: len(sort_cols)]
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def train_baseline_models(
    task: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Antreneaza toate modelele de baza si intoarce tabelul cu metrici."""
    models = get_models(task, random_state)
    trained_models: Dict[str, Pipeline] = {}
    rows: List[Dict[str, Any]] = []

    for name, model in models.items():
        print(f"[Baseline] Antrenez: {name}")
        estimator = make_pipeline(X_train, clone(model))
        estimator.fit(X_train, y_train)
        metrics = evaluate_model(task, estimator, X_test, y_test)
        rows.append({"Model": name, **metrics})
        trained_models[name] = estimator

    results = pd.DataFrame(rows)
    results = rank_results(task, results)
    return results, trained_models


def get_param_search_spaces(task: str) -> Dict[str, Dict[str, Any]]:
    """
    Spatiile de cautare pentru toate modelele.
    method='grid' foloseste GridSearchCV.
    method='bayes' foloseste BayesSearchCV daca scikit-optimize este instalat.
    """
    if task == "classification":
        spaces = {
            "Naive Bayes": {
                "method": "grid",
                "params": {"model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
            },
            "Logistic Regression": {
                "method": "grid",
                "params": {
                    "model__C": [0.01, 0.1, 1, 10, 100],
                    "model__penalty": ["l2"],
                    "model__solver": ["lbfgs", "liblinear"],
                },
            },
            "Decision Tree": {
                "method": "grid",
                "params": {
                    "model__max_depth": [None, 3, 5, 8, 12],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__criterion": ["gini", "entropy"],
                },
            },
            "Random Forest": {
                "method": "bayes",
                "params": {
                    "model__n_estimators": Integer(100, 450) if Integer else [100, 200, 350],
                    "model__max_depth": Integer(2, 25) if Integer else [None, 5, 10, 20],
                    "model__min_samples_split": Integer(2, 15) if Integer else [2, 5, 10],
                    "model__min_samples_leaf": Integer(1, 8) if Integer else [1, 2, 4],
                    "model__max_features": Categorical(["sqrt", "log2", None]) if Categorical else ["sqrt", "log2", None],
                },
            },
            "Support Vector Machine": {
                "method": "bayes",
                "params": {
                    "model__C": Real(1e-2, 1e2, prior="log-uniform") if Real else [0.1, 1, 10, 100],
                    "model__gamma": Real(1e-4, 1.0, prior="log-uniform") if Real else ["scale", 0.01, 0.1, 1],
                    "model__kernel": Categorical(["rbf", "linear"]) if Categorical else ["rbf", "linear"],
                    "model__probability": Categorical([True]) if Categorical else [True],
                },
            },
            "K-Nearest Neighbors": {
                "method": "grid",
                "params": {
                    "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                },
            },
            "XGBoost": {
                "method": "bayes",
                "params": {
                    "model__n_estimators": Integer(80, 450) if Integer else [100, 200, 350],
                    "model__max_depth": Integer(2, 8) if Integer else [2, 4, 6],
                    "model__learning_rate": Real(0.01, 0.30, prior="log-uniform") if Real else [0.03, 0.1, 0.2],
                    "model__subsample": Real(0.60, 1.00) if Real else [0.7, 0.9, 1.0],
                    "model__colsample_bytree": Real(0.60, 1.00) if Real else [0.7, 0.9, 1.0],
                },
            },
            "CatBoost": {
                "method": "bayes",
                "params": {
                    "model__iterations": Integer(100, 500) if Integer else [150, 300, 450],
                    "model__depth": Integer(3, 8) if Integer else [3, 5, 7],
                    "model__learning_rate": Real(0.01, 0.30, prior="log-uniform") if Real else [0.03, 0.1, 0.2],
                    "model__l2_leaf_reg": Real(1.0, 10.0) if Real else [1, 3, 7],
                },
            },
            "Explainable Boosting Machine": {
                "method": "bayes",
                "params": {
                    "model__max_bins": Integer(64, 256) if Integer else [64, 128, 256],
                    "model__interactions": Integer(0, 10) if Integer else [0, 5, 10],
                    "model__learning_rate": Real(0.005, 0.05, prior="log-uniform") if Real else [0.005, 0.01, 0.03],
                    "model__max_leaves": Integer(2, 5) if Integer else [2, 3, 5],
                },
            },
        }
        return spaces

    spaces = {
        "Linear Regression": {
            "method": "grid",
            "params": {"model__fit_intercept": [True, False]},
        },
        "Decision Tree Regressor": {
            "method": "grid",
            "params": {
                "model__max_depth": [None, 3, 5, 8, 12],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest Regressor": {
            "method": "bayes",
            "params": {
                "model__n_estimators": Integer(100, 450) if Integer else [100, 200, 350],
                "model__max_depth": Integer(2, 25) if Integer else [None, 5, 10, 20],
                "model__min_samples_split": Integer(2, 15) if Integer else [2, 5, 10],
                "model__min_samples_leaf": Integer(1, 8) if Integer else [1, 2, 4],
                "model__max_features": Categorical(["sqrt", "log2", None]) if Categorical else ["sqrt", "log2", None],
            },
        },
        "Support Vector Regressor": {
            "method": "bayes",
            "params": {
                "model__C": Real(1e-1, 1e3, prior="log-uniform") if Real else [1, 10, 100],
                "model__gamma": Real(1e-4, 1.0, prior="log-uniform") if Real else ["scale", 0.01, 0.1],
                "model__epsilon": Real(0.01, 1.0, prior="log-uniform") if Real else [0.01, 0.1, 0.5],
                "model__kernel": Categorical(["rbf", "linear"]) if Categorical else ["rbf", "linear"],
            },
        },
        "K-Nearest Neighbor Regressor": {
            "method": "grid",
            "params": {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "Gaussian Process Regressor": {
            "method": "grid",
            "params": {
                "model__alpha": [1e-10, 1e-5, 1e-3, 1e-2],
                "model__normalize_y": [True, False],
            },
        },
        "XGBoost Regressor": {
            "method": "bayes",
            "params": {
                "model__n_estimators": Integer(80, 500) if Integer else [100, 250, 400],
                "model__max_depth": Integer(2, 8) if Integer else [2, 4, 6],
                "model__learning_rate": Real(0.01, 0.30, prior="log-uniform") if Real else [0.03, 0.1, 0.2],
                "model__subsample": Real(0.60, 1.00) if Real else [0.7, 0.9, 1.0],
                "model__colsample_bytree": Real(0.60, 1.00) if Real else [0.7, 0.9, 1.0],
            },
        },
        "CatBoost Regressor": {
            "method": "bayes",
            "params": {
                "model__iterations": Integer(100, 500) if Integer else [150, 300, 450],
                "model__depth": Integer(3, 8) if Integer else [3, 5, 7],
                "model__learning_rate": Real(0.01, 0.30, prior="log-uniform") if Real else [0.03, 0.1, 0.2],
                "model__l2_leaf_reg": Real(1.0, 10.0) if Real else [1, 3, 7],
            },
        },
        "Explainable Boosting Regressor": {
            "method": "bayes",
            "params": {
                "model__max_bins": Integer(64, 256) if Integer else [64, 128, 256],
                "model__interactions": Integer(0, 10) if Integer else [0, 5, 10],
                "model__learning_rate": Real(0.005, 0.05, prior="log-uniform") if Real else [0.005, 0.01, 0.03],
                "model__max_leaves": Integer(2, 5) if Integer else [2, 3, 5],
            },
        },
    }
    return spaces


def bayes_space_to_small_grid(space: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Fallback daca skopt nu este instalat."""
    grid: Dict[str, List[Any]] = {}
    for key, value in space.items():
        if isinstance(value, list):
            grid[key] = value[:3]
        else:
            if "n_estimators" in key or "iterations" in key:
                grid[key] = [100, 250]
            elif "max_depth" in key or "depth" in key:
                grid[key] = [3, 6]
            elif "learning_rate" in key:
                grid[key] = [0.03, 0.1]
            elif "C" in key:
                grid[key] = [1, 10]
            elif "gamma" in key:
                grid[key] = ["scale", 0.1]
            elif "kernel" in key:
                grid[key] = ["rbf", "linear"]
            elif "probability" in key:
                grid[key] = [True]
            elif "subsample" in key or "colsample" in key:
                grid[key] = [0.8, 1.0]
            elif "max_features" in key:
                grid[key] = ["sqrt", None]
            elif "interactions" in key:
                grid[key] = [0, 5]
            elif "max_bins" in key:
                grid[key] = [64, 128]
            elif "max_leaves" in key:
                grid[key] = [2, 3]
            else:
                grid[key] = [None]
    return grid

 
def tune_all_models(
    task: str,
    baseline_estimators: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: int = 3,
    n_iter_bayes: int = 12,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """
    Ajusteaza hiperparametrii pentru TOTI algoritmii, nu doar pentru primii 5.
    Returneaza tabelul de rezultate, modelele ajustate si hiperparametrii gasiti.
    """
    spaces = get_param_search_spaces(task)
    scoring = "f1" if task == "classification" else "neg_root_mean_squared_error"

    tuned_estimators: Dict[str, Pipeline] = {}
    best_params: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for name, base_estimator in baseline_estimators.items():
        print(f"[Tuning] Ajustez: {name}")
        estimator = clone(base_estimator)
        spec = spaces.get(name, {"method": "grid", "params": {}})
        params = spec.get("params", {})
        method = spec.get("method", "grid")

        if not params:
            estimator.fit(X_train, y_train)
            best_model = estimator
            params_found = {}
        else:
            try:
                if method == "bayes" and BayesSearchCV is not None:
                    search = BayesSearchCV(
                        estimator=estimator,
                        search_spaces=params,
                        n_iter=n_iter_bayes,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=1,
                        random_state=random_state,
                        error_score=np.nan,
                        refit=True,
                    )
                else:
                    if method == "bayes":
                        params = bayes_space_to_small_grid(params)
                    search = GridSearchCV(
                        estimator=estimator,
                        param_grid=params,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1,
                        error_score=np.nan,
                        refit=True,
                    )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                params_found = search.best_params_
            except Exception as exc:
                print(f"[WARN] Tuning esuat pentru {name}: {exc}. Folosesc modelul baseline reantrenat.")
                estimator.fit(X_train, y_train)
                best_model = estimator
                params_found = {"warning": str(exc)}

        metrics = evaluate_model(task, best_model, X_test, y_test)
        rows.append({"Model": name, **metrics})
        tuned_estimators[name] = best_model
        best_params[name] = {k: str(v) for k, v in params_found.items()}

    results = pd.DataFrame(rows)
    results = rank_results(task, results)
    return results, tuned_estimators, best_params
 

def save_eda_plots(df: pd.DataFrame, target: str, task: str, output_dir: str | Path) -> Dict[str, str]:
    """Genereaza si salveaza grafice EDA relevante."""
    output_dir = ensure_dir(output_dir)
    paths: Dict[str, str] = {}

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    plt.figure(figsize=(7, 4))
    if task == "classification":
        counts = df[target].value_counts().sort_index()
        plt.bar([str(x) for x in counts.index], counts.values)
        plt.xlabel("Clasa")
        plt.ylabel("Numar observatii")
        plt.title(f"Distributia variabilei tinta: {target}")
    else:
        plt.hist(df[target], bins=30)
        plt.xlabel(target)
        plt.ylabel("Frecventa")
        plt.title(f"Distributia variabilei tinta: {target}")
    plt.tight_layout()
    path = output_dir / "01_target_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["target_distribution"] = str(path)

    features_num = [c for c in numeric_cols if c != target]
    if features_num:
        n = len(features_num)
        cols = 3
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, features_num):
            ax.hist(df[col].dropna(), bins=25)
            ax.set_title(f"Distributie: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecventa")
        for ax in axes[len(features_num):]:
            ax.axis("off")
        plt.tight_layout()
        path = output_dir / "02_numeric_distributions.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths["numeric_distributions"] = str(path)

    if categorical_cols:
        n = len(categorical_cols)
        cols = 2
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, categorical_cols):
            counts = df[col].astype(str).value_counts()
            ax.bar(counts.index, counts.values)
            ax.set_title(f"Frecvente: {col}")
            ax.tick_params(axis="x", rotation=35)
        for ax in axes[len(categorical_cols):]:
            ax.axis("off")
        plt.tight_layout()
        path = output_dir / "03_categorical_frequencies.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths["categorical_frequencies"] = str(path)

    corr_cols = [c for c in numeric_cols]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(1.0 * len(corr_cols) + 4, 1.0 * len(corr_cols) + 3))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(corr_cols)))
        ax.set_yticklabels(corr_cols)
        fig.colorbar(im, ax=ax)
        ax.set_title("Matricea de corelatie pentru variabile numerice")
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        path = output_dir / "04_correlation_matrix.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths["correlation_matrix"] = str(path)

    rel_cols = [c for c in features_num[:6]]
    if rel_cols:
        n = len(rel_cols)
        cols = 2
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, rel_cols):
            if task == "classification":
                groups = [df.loc[df[target] == cls, col].dropna() for cls in sorted(df[target].unique())]
                ax.boxplot(groups, labels=[str(cls) for cls in sorted(df[target].unique())])
                ax.set_xlabel(target)
                ax.set_ylabel(col)
                ax.set_title(f"{col} in functie de clasa")
            else:
                ax.scatter(df[col], df[target], alpha=0.45)
                ax.set_xlabel(col)
                ax.set_ylabel(target)
                ax.set_title(f"Relatie {col} - {target}")
        for ax in axes[len(rel_cols):]:
            ax.axis("off")
        plt.tight_layout()
        path = output_dir / "05_numeric_target_relationships.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths["numeric_target_relationships"] = str(path)

    with open(output_dir / "eda_plot_paths.json", "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)
    return paths

def plot_learning_curve_for_model(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    title: str,
    cv: int = 3,
) -> plt.Figure:
    """Genereaza figura pentru curba de invatare a unui model."""
    scoring = "f1" if task == "classification" else "neg_root_mean_squared_error"
    train_sizes = np.linspace(0.2, 1.0, 5)

    sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    if task == "regression":
        train_mean = -train_mean
        valid_mean = -valid_mean
        ylabel = "RMSE"
    else:
        ylabel = "F1-score"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, train_mean, marker="o", label="Scor antrenare")
    ax.plot(sizes, valid_mean, marker="o", label="Scor validare")
    ax.set_title(title)
    ax.set_xlabel("Numar exemple de antrenare")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def save_learning_curves(
    task: str,
    estimators: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    cv: int = 3,
) -> Dict[str, str]:
    """Salveaza curbele de invatare pentru toti algoritmii ajustati."""
    output_dir = ensure_dir(output_dir)
    paths: Dict[str, str] = {}
    for name, estimator in estimators.items():
        print(f"[Learning curve] {name}")
        try:
            fig = plot_learning_curve_for_model(estimator, X, y, task, f"Curba de invatare - {name}", cv=cv)
            path = output_dir / f"learning_curve_{safe_name(name)}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths[name] = str(path)
        except Exception as exc:
            print(f"[WARN] Nu am putut genera curba pentru {name}: {exc}")
    with open(output_dir / "learning_curve_paths.json", "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)
    return paths

def _model_prediction_function(model: Any, task: str):
    """Functie de predictie folosita de SHAP pe date deja transformate."""

    def predict_fn(X_array: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X_array)
        if task == "classification":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_array)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]
                return proba.ravel()
            if hasattr(model, "decision_function"):
                return model.decision_function(X_array)
            return model.predict(X_array)
        return model.predict(X_array)

    return predict_fn


def compute_shap_explanation(
    pipeline: Pipeline,
    X_background_raw: pd.DataFrame,
    X_explain_raw: pd.DataFrame,
    task: str,
    max_background: int = 60,
) -> Tuple[Any, np.ndarray, List[str]]:
    """
    Calculeaza SHAP model-agnostic, deci functioneaza pentru toti algoritmii.
    Pentru viteza, foloseste un esantion de background.
    """
    if shap is None:
        raise ImportError("Pachetul shap nu este instalat. Ruleaza: pip install shap")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_bg = X_background_raw.sample(min(max_background, len(X_background_raw)), random_state=RANDOM_STATE)
    X_bg_t = np.asarray(preprocessor.transform(X_bg))
    X_exp_t = np.asarray(preprocessor.transform(X_explain_raw))
    feature_names = get_feature_names_from_pipeline(pipeline)

    predict_fn = _model_prediction_function(model, task)
    masker = shap.maskers.Independent(X_bg_t)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation", feature_names=feature_names)

    max_evals = max(2 * X_exp_t.shape[1] + 1, 100)
    explanation = explainer(X_exp_t, max_evals=max_evals)
    return explanation, X_exp_t, feature_names


def _new_fig() -> plt.Figure:
    plt.close("all")
    return plt.figure()


def save_shap_plots_for_model(
    task: str,
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str | Path,
    sample_size: int = 80,
) -> Dict[str, str]:
    """
    Genereaza SHAP pentru un model:
    - summary plot
    - bar plot
    - waterfall plot pentru o predictie
    - force plot pentru o predictie
    - scatter plots pentru primele 3 caracteristici importante
    """
    if shap is None:
        raise ImportError("Pachetul shap nu este instalat.")

    output_dir = ensure_dir(output_dir)
    paths: Dict[str, str] = {}

    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    explanation, X_sample_t, feature_names = compute_shap_explanation(
        pipeline,
        X_train,
        X_sample,
        task,
        max_background=min(60, len(X_train)),
    )

    base = safe_name(name)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(explanation.values, X_sample_t, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    path = output_dir / f"shap_summary_{base}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    paths["summary"] = str(path)

    plt.figure(figsize=(9, 6))
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    path = output_dir / f"shap_bar_{base}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    paths["bar"] = str(path)

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    plt.tight_layout()
    path = output_dir / f"shap_waterfall_{base}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    paths["waterfall"] = str(path)

    try:
        plt.figure(figsize=(12, 3.5))
        shap.force_plot(
            explanation.base_values[0],
            explanation.values[0],
            X_sample_t[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        path = output_dir / f"shap_force_{base}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        paths["force"] = str(path)
    except Exception as exc:
        print(f"[WARN] Force plot esuat pentru {name}: {exc}")

    mean_abs = np.abs(explanation.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:3]
    scatter_paths = []
    for idx in top_idx:
        try:
            plt.figure(figsize=(7, 5))
            shap.plots.scatter(explanation[:, idx], show=False)
            plt.tight_layout()
            scatter_path = output_dir / f"shap_scatter_{base}_{safe_name(feature_names[idx])}.png"
            plt.savefig(scatter_path, dpi=160, bbox_inches="tight")
            plt.close()
            scatter_paths.append(str(scatter_path))
        except Exception as exc:
            print(f"[WARN] Scatter plot esuat pentru {name}, feature {feature_names[idx]}: {exc}")
    paths["scatter"] = scatter_paths

    top_features = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in top_idx
    ]
    with open(output_dir / f"shap_top_features_{base}.json", "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2, ensure_ascii=False)
    paths["top_features_json"] = str(output_dir / f"shap_top_features_{base}.json")

    return paths


def save_shap_plots_all_models(
    task: str,
    estimators: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str | Path,
    sample_size: int = 80,
) -> Dict[str, Dict[str, str]]:
    """Genereaza SHAP pentru toti algoritmii ajustati."""
    output_dir = ensure_dir(output_dir)
    all_paths: Dict[str, Dict[str, str]] = {}
    for name, estimator in estimators.items():
        print(f"[SHAP] {name}")
        try:
            all_paths[name] = save_shap_plots_for_model(
                task,
                name,
                estimator,
                X_train,
                X_test,
                output_dir,
                sample_size=sample_size,
            )
        except Exception as exc:
            print(f"[WARN] SHAP esuat pentru {name}: {exc}")
            all_paths[name] = {"warning": str(exc)}
    with open(output_dir / "shap_plot_paths.json", "w", encoding="utf-8") as f:
        json.dump(all_paths, f, indent=2, ensure_ascii=False)
    return all_paths


def make_local_shap_figures_for_input(
    task: str,
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_input: pd.DataFrame,
) -> Dict[str, plt.Figure]:
    """Genereaza figuri SHAP locale pentru Streamlit."""
    if shap is None:
        raise ImportError("Pachetul shap nu este instalat.")

    explanation, X_input_t, feature_names = compute_shap_explanation(
        pipeline,
        X_background,
        X_input,
        task,
        max_background=min(60, len(X_background)),
    )
    figs: Dict[str, plt.Figure] = {}

    
    plt.close("all")
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    figs["waterfall"] = plt.gcf()

    values = explanation.values[0]
    order = np.argsort(np.abs(values))[::-1][:12]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [feature_names[i] for i in order][::-1]
    vals = [values[i] for i in order][::-1]
    ax.barh(labels, vals)
    ax.axvline(0, linewidth=1)
    ax.set_title("Contributii SHAP locale pentru predictia curenta")
    ax.set_xlabel("Valoare SHAP")
    fig.tight_layout()
    figs["local_bar"] = fig

    return figs


def save_artifacts(
    task: str,
    output_root: str | Path,
    baseline_results: pd.DataFrame,
    tuned_results: pd.DataFrame,
    tuned_estimators: Dict[str, Pipeline],
    best_params: Dict[str, Dict[str, Any]],
    data_path: str | Path,
    target: str,
) -> None:
    """Salveaza modelele, metricile si metadatele pentru Streamlit."""
    task_dir = ensure_dir(Path(output_root) / task)
    models_dir = ensure_dir(task_dir / "models")

    baseline_results.to_csv(task_dir / "baseline_metrics.csv", index=False)
    tuned_results.to_csv(task_dir / "tuned_metrics.csv", index=False)

    for name, estimator in tuned_estimators.items():
        joblib.dump(estimator, models_dir / f"{safe_name(name)}.joblib")

    with open(task_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    metadata = {
        "task": task,
        "target": target,
        "data_path": str(data_path),
        "models": {name: f"models/{safe_name(name)}.joblib" for name in tuned_estimators},
        "ranking_metric": "F1, ROC_AUC, Accuracy" if task == "classification" else "RMSE, MAE, R2",
    }
    with open(task_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_artifacts(task: str, output_root: str | Path = "outputs") -> Dict[str, Any]:
    """Incarca artefactele salvate dupa rularea notebook-urilor sau train_all.py."""
    task_dir = Path(output_root) / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Nu exista folderul {task_dir}. Ruleaza notebook-ul sau python src/train_all.py.")

    with open(task_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    models = {}
    for name, rel_path in metadata["models"].items():
        models[name] = joblib.load(task_dir / rel_path)

    tuned_metrics = pd.read_csv(task_dir / "tuned_metrics.csv")
    baseline_metrics = pd.read_csv(task_dir / "baseline_metrics.csv")

    params_path = task_dir / "best_params.json"
    best_params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}

    return {
        "metadata": metadata,
        "models": models,
        "tuned_metrics": tuned_metrics,
        "baseline_metrics": baseline_metrics,
        "best_params": best_params,
        "task_dir": task_dir,
    }


def execute_complete_pipeline(
    task: str,
    csv_path: str | Path,
    target: str,
    output_root: str | Path = "outputs",
    top_n: int = 5,
    cv: int = 3,
    n_iter_bayes: int = 12,
    shap_sample_size: int = 80,
) -> Dict[str, Any]:
    """
    Rulează pipeline-ul complet: EDA, Baseline, extrage Top N modele, 
    face Hyperparameter Tuning, Learning Curves și SHAP doar pe ele.
    """
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = split_data(df, target, task)

    task_dir = ensure_dir(Path(output_root) / task)
    eda_dir = ensure_dir(task_dir / "figures" / "eda")
    curves_dir = ensure_dir(task_dir / "figures" / "learning_curves")
    shap_dir = ensure_dir(task_dir / "figures" / "shap")

    print("[EDA] Generez grafice exploratorii...")
    eda_paths = save_eda_plots(df, target, task, eda_dir)

    print("[Baseline] Antrenez modelele de baza...")
    baseline_results, baseline_estimators = train_baseline_models(task, X_train, X_test, y_train, y_test)

    top_models_names = baseline_results['Model'].head(top_n).tolist()
    top_estimators = {name: baseline_estimators[name] for name in top_models_names}

    print("[Tuning] hiperparametri pentru Top {top_n}: {top_models_names}...")
    tuned_results, tuned_estimators, best_params = tune_all_models(
        task,
        top_estimators,
        X_train,
        X_test,
        y_train,
        y_test,
        cv=cv,
        n_iter_bayes=n_iter_bayes,
    )

    print("[Learning curves] Generez curbe...")
    X_all, y_all = pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
    learning_paths = save_learning_curves(task, tuned_estimators, X_all, y_all, curves_dir, cv=cv)

    print("[SHAP] Generez grafice SHAP...")
    shap_paths = save_shap_plots_all_models(task, tuned_estimators, X_train, X_test, shap_dir, sample_size=shap_sample_size)

    save_artifacts(task, output_root, baseline_results, tuned_results, tuned_estimators, best_params, csv_path, target)

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "baseline_results": baseline_results,
        "tuned_results": tuned_results,
        "tuned_estimators": tuned_estimators,
        "best_params": best_params,
        "eda_paths": eda_paths,
        "learning_paths": learning_paths,
        "shap_paths": shap_paths,
    }
