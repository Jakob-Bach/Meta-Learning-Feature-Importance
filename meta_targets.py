"""Meta-targets

Module to compute meta-targets, i.e., feature importance regarding to different base models
and feature importance-measures.
"""

import sklearn.ensemble
import sklearn.linear_model
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
