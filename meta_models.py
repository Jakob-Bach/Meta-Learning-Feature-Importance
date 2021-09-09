"""Meta-models

Module to train and evaluate meta-models, i.e., regression models that predict feature importance.
"""

from typing import Any, Callable, Dict

import pandas as pd
import sklearn.base
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree


META_MODELS = {
    'Decision tree': {'func': sklearn.tree.DecisionTreeRegressor,
                      'args': {'random_state': 25}},
    'kNN': {'func': sklearn.neighbors.KNeighborsRegressor, 'args': {}},
    'Linear regression': {'func': sklearn.linear_model.LinearRegression,
                          'args': {'random_state': 25}},
    'Random forest': {'func': sklearn.ensemble.RandomForestRegressor,
                      'args': {'random_state': 25}},
    'SVM': {'func': sklearn.svm.SVR, 'args': {'random_state': 25}}
}


# In a cross-validation procedure, train the model defined by "model_func" and "model_args" with
# the meta-features "X" and the meta-target "y". Evaluate model performance and return these
# performance values (one cross-validation fold corresponds to one row in the resulting table).
def predict_and_evaluate(
        X: pd.DataFrame, y: pd.Series, model_func: Callable[..., sklearn.base.BaseEstimator],
        model_args: Dict[str, Any]) -> pd.DataFrame:
    scaler = sklearn.preprocessing.StandardScaler()
    splitter = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=25)
    prediction_results = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X=X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        X_train = pd.DataFrame(scaler.fit_transform(X=X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X=X_test), columns=X_test.columns)
        model = model_func(**model_args)
        model.fit(X=X_train, y=y_train)
        train_score = sklearn.metrics.r2_score(y_true=y_train, y_pred=model.predict(X=X_train))
        test_score = sklearn.metrics.r2_score(y_true=y_test, y_pred=model.predict(X=X_test))
        result = {'fold_id': fold_id, 'train_score': train_score, 'test_score': test_score}
        prediction_results.append(result)
    return pd.DataFrame(prediction_results)
