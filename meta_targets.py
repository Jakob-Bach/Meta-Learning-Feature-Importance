"""Meta-targets

Module to compute meta-targets, i.e., feature importance regarding to different base models
and feature importance-measures.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Collection, Dict
import warnings

import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
import sklearn.base
import sklearn.ensemble
import sklearn.inspection
import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.tree


BASE_MODELS = {
    'Decision tree': {'func': sklearn.tree.DecisionTreeClassifier,
                      'args': {'random_state': 25}},
    'kNN': {'func': sklearn.neighbors.KNeighborsClassifier, 'args': {}},
    'Logistic regression': {'func': sklearn.linear_model.LogisticRegression,
                            'args': {'random_state': 25}},
    'Naive Bayes': {'func': sklearn.naive_bayes.GaussianNB, 'args': {}},
    'Random forest': {'func': sklearn.ensemble.RandomForestClassifier,
                      'args': {'random_state': 25}},
    'SVM': {'func': sklearn.svm.SVC,
            'args': {'random_state': 25, 'probability': True}}  # LIME uses probabilities
}

BASE_METRIC = sklearn.metrics.matthews_corrcoef


class FeatureImportanceMeasure(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def compute_importance_with_model(X: pd.DataFrame, y: pd.Series,
                                      model: sklearn.base.BaseEstimator) -> Collection[float]:
        raise NotImplementedError('Abstract method.')

    # Compute (model-based) feature importance for model defined by "model_func" and "model_args",
    # trained on dataset consisting of features values "X" and target "y". Return importance
    # (a float value) for each feature (column) of "X". Note that importances are not normalized
    # (e.g., to [0,1] or [0,100]) and might also be negative.
    @classmethod
    def compute_importance(cls, X: pd.DataFrame, y: pd.Series,
                           model_func: Callable[..., sklearn.base.BaseEstimator],
                           model_args: Dict[str, Any]) -> Collection[float]:
        scaler = sklearn.preprocessing.StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X=X), columns=X.columns)
        model = model_func(**model_args)
        model.fit(X=X, y=y)
        return cls.compute_importance_with_model(X=X, y=y, model=model)


class DropColumnImportance(FeatureImportanceMeasure):

    # Re-train model with data where individual features are missing and compare to prediction
    # quality with original data.
    @staticmethod
    def compute_importance_with_model(X: pd.DataFrame, y: pd.Series,
                                      model: sklearn.base.BaseEstimator) -> Collection[float]:
        importances = []
        baseline = BASE_METRIC(y_true=y, y_pred=model.predict(X=X))
        model = sklearn.base.clone(model)  # do not modify original model when re-training
        for feature in X.columns:
            X_temp = X.drop(columns=feature)
            model.fit(X=X_temp, y=y)
            importances.append(baseline - BASE_METRIC(y_true=y, y_pred=model.predict(X_temp)))
        return importances


class LIMEImportance(FeatureImportanceMeasure):

    # Train linear model with data objects from neighborhood of object under discussion (create
    # them by perturbation) and extract coefficients from that explanation model (which means that
    # features can have positive as well as negative impact on target values). To get global
    # values for feature importance, repeat for / average over all data objects in "X".
    @staticmethod
    def compute_importance_with_model(X: pd.DataFrame, y: pd.Series,
                                      model: sklearn.base.BaseEstimator) -> Collection[float]:
        # Expanation is slower with discretization, so we disable the latter and use numeric
        # feature as they are, sampling in their neighborhood (default: sample around global mean):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values, mode='classification', feature_names=X.columns,
            discretize_continuous=False, random_state=25, sample_around_instance=True)
        importances = [pd.DataFrame(explainer.explain_instance(
            data_row=row.values, predict_fn=model.predict_proba, labels=[1],
            num_features=X.shape[1], num_samples=100).as_list(), columns=['Feature', 'Importance'])
            for _, row in X.iterrows()]
        importances = pd.concat(importances)
        importances = importances.groupby('Feature')['Importance'].mean().reset_index()
        # Make sure importances are ordered as the features in the original dataset:
        return pd.DataFrame({'Feature': X.columns}).merge(importances)['Importance'].values


class PermutationImportance(FeatureImportanceMeasure):

    # Make predictions with data where individual features are permuted and compare to prediction
    # quality with original data. No re-training of model necessary.
    @staticmethod
    def compute_importance_with_model(X: pd.DataFrame, y: pd.Series,
                                      model: sklearn.base.BaseEstimator) -> Collection[float]:
        with warnings.catch_warnings():  # warning if y_true or y_pred is constant in MCC call
            warnings.filterwarnings(action='ignore',
                                    message='invalid value encountered in double_scalars')
            result = sklearn.inspection.permutation_importance(
                estimator=model, X=X, y=y, n_repeats=5, random_state=25,
                scoring=sklearn.metrics.make_scorer(BASE_METRIC))
            return result['importances_mean']  # permutation is repeated and results are averaged


class SHAPImportance(FeatureImportanceMeasure):

    # Make predictions with different subsets of features, replacing remaining ones with expected
    # value, i.e., compute feature contributions (which can be positive and negative) to different
    # "coalitions", similar to the game-theoretic concept of Shapley values. No re-training of
    # model necessary.
    @staticmethod
    def compute_importance_with_model(X: pd.DataFrame, y: pd.Series,
                                      model: sklearn.base.BaseEstimator) -> Collection[float]:
        np.random.seed(25)  # unfortunately, we cannot set a seed for the explanation directly
        explainer = shap.Explainer(
            algorithm='auto', model=lambda data: model.predict_proba(X=data)[:, 1],
            masker=shap.maskers.Independent(data=X, max_samples=X.shape[0]))  # do not sample
        explanation = explainer(X, silent=True)  # Explainer object is a callable
        return explanation.values.mean(axis=0)


IMPORTANCE_MEASURES = {'Drop-column': DropColumnImportance, 'LIME': LIMEImportance,
                       'Permutation': PermutationImportance, 'SHAP': SHAPImportance}
