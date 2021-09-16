"""Meta-features

Module to compute meta-features.
"""

import pandas as pd
import pymfe.mfe


# Compute datasets and feature-meta-features for a base datasets consisting of a feature part "X"
# and a target part X. Return meta-feature table, with rows corresponding to base features and
# columns corresponding to meta-features.
def compute_meta_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # Dataset meta-features:
    mfe = pymfe.mfe.MFE(groups=['general', 'statistical', 'info-theory'], summary=['mean', 'sd'],
                        random_state=25)  # some of the meta-features involve randomness
    mfe.fit(X=X.values, y=y.values, suppress_warnings=True)
    data_meta_features = mfe.extract(suppress_warnings=True, out_type=pd.DataFrame)  # just one row
    data_meta_features.rename(columns=lambda x: 'data.' + x, inplace=True)

    # Feature meta-features ("summary" here effectively is identity function, as just one feature):
    mfe = pymfe.mfe.MFE(groups=['general', 'statistical', 'info-theory'], summary='mean',
                        random_state=25)
    feature_meta_features = []
    for feature in X.columns:
        mfe.fit(X=X[feature].values, y=y.values, suppress_warnings=True)
        feature_meta_features.append(mfe.extract(suppress_warnings=True, out_type=pd.DataFrame))
    feature_meta_features = pd.concat(feature_meta_features)
    feature_meta_features.rename(columns=lambda x: 'feature.' + x.replace('.mean', ''), inplace=True)

    return pd.concat([feature_meta_features, data_meta_features], axis='columns')
