"""Prepare datasets

Script that:
- downloads, pre-processes, and saves base datasets from OpenML
- computes meta-features
- computes meta-targets (combining feature-importance measures and base models)
- saves the meta-datasets

Usage: python -m prepare_datasets --help
"""

import argparse
import pathlib
from typing import Optional, Sequence
import warnings

import numpy as np
import openml
import pandas as pd
import sklearn.preprocessing
import tqdm

import data_utility


# Download one base dataset with the given "data_id" from OpenML and store it in X, y format in
# "base_data_dir", all columns made numeric. Note that the method might throw an exception if
# OpenML is not able to retrieve the dataset.
def download_base_dataset(data_id: int, base_data_dir: pathlib.Path) -> None:
    dataset = openml.datasets.get_dataset(dataset_id=data_id, download_data=True)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    non_numeric_features = [x.name for x in dataset.features.values()
                            if (x.name in X.columns) and (x.data_type != 'numeric')]
    X[non_numeric_features] = sklearn.preprocessing.OrdinalEncoder(dtype=int).fit_transform(
        X=X[non_numeric_features])
    assert all(np.issubdtype(X[feature].dtype, np.number) for feature in X.columns)
    y = pd.Series(sklearn.preprocessing.LabelEncoder().fit_transform(y=y), name=y.name)
    data_utility.save_dataset(X, y, dataset_name=dataset.name, directory=base_data_dir)


# Download OpenML datasets and store them in "base_data_dir". Either retrieve base datasets by
# "data_ids" or search according to fixed dataset characteristics. The latter was done for the
# paper, but the datasets matching the characteristics can change in future.
def prepare_base_datasets(base_data_dir: pathlib.Path, data_ids: Optional[Sequence[int]] = None) -> None:
    print('Base dataset preparation started.')
    if not base_data_dir.is_dir():
        print('Base-dataset directory does not exist. We create it.')
        base_data_dir.mkdir(parents=True)
    if any(base_data_dir.iterdir()):
        print('Base-dataset directory is not empty. Files might be overwritten, but not deleted.')
    dataset_overview = openml.datasets.list_datasets(status='active', output_format='dataframe')
    if (data_ids is None) or (len(data_ids) == 0):
        dataset_overview = dataset_overview[
            (dataset_overview['NumberOfClasses'] == 2) &  # binary classification
            (dataset_overview['NumberOfInstances'] >= 1000) &
            (dataset_overview['NumberOfInstances'] <= 10000) &
            (dataset_overview['NumberOfMissingValues'] == 0)
        ]
        # Pick latest version of each dataset:
        dataset_overview = dataset_overview.sort_values(by='version').groupby('name').last().reset_index()
        # Pick the same amount of datasets from different categories regarding number of features:
        feature_number_groups = [(6, 11), (12, 26), (27, 51)]  # list of (lower, upper); count includes target
        num_datasets_per_group = 20
        data_ids = []
        with tqdm.tqdm(total=(len(feature_number_groups) * num_datasets_per_group),
                       desc='Downloading datasets') as progress_bar:
            for lower, upper in feature_number_groups:
                current_datasets = dataset_overview[(dataset_overview['NumberOfFeatures'] >= lower) &
                                                    (dataset_overview['NumberOfFeatures'] <= upper)]
                successful_downloads = 0
                current_position = 0  # ... in the table of datasets
                while successful_downloads < num_datasets_per_group:
                    data_id = int(current_datasets['did'].iloc[current_position])
                    try:
                        download_base_dataset(data_id=data_id, base_data_dir=base_data_dir)
                        data_ids.append(data_id)
                        successful_downloads += 1
                        progress_bar.update()
                    except Exception:  # OpenML does not specify exception type for get_dataset()
                        pass
                    finally:  # in any case, move on to next dataset
                        current_position += 1
    else:
        print('Using given dataset ids.')
        for data_id in tqdm.tqdm(data_ids, desc='Downloading datasets'):
            try:
                download_base_dataset(data_id=data_id, base_data_dir=base_data_dir)
            except Exception:  # OpenML does not specify exception type for get_dataset()
                warnings.warn(f'Download of dataset {data_id} failed.')
    dataset_overview[dataset_overview['did'].isin(data_ids)].to_csv(
        base_data_dir / '_dataset_overview.csv', index=False)
    print('Base datasets prepared and saved.')


# For each base dataset from "base_data_dir", compute all meta-features and meta-targets, i.e.,
# all feature-importance measures for all base models. Save the resulting meta-datasets into
# "meta_data_dir".
def prepare_meta_datasets(base_data_dir: pathlib.Path, meta_data_dir: pathlib.Path) -> None:
    print('Meta-dataset preparation started.')
    print('Meta-datasets prepared and saved.')


# Parse command-line arguments and prepare base + meta datasets.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves base datasets from OpenML, creates meta-datasets ' +
        'and stores all these data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--base_data_dir', type=pathlib.Path, default='data/base_datasets/',
                        help='Directory to store base datasets. Will be created if necessary.')
    parser.add_argument('-i', '--data_ids', type=int, default=[], nargs='*',
                        help='Ids of OpenML datasets. If none provided, will search for datasets.')
    parser.add_argument('-m', '--meta_data_dir', type=pathlib.Path, default='data/meta_datasets/',
                        help='Directory to store meta-datasets. Will be created if necessary.')
    args = parser.parse_args()
    prepare_base_datasets(base_data_dir=args.base_data_dir, data_ids=args.data_ids)
    prepare_meta_datasets(base_data_dir=args.base_data_dir, meta_data_dir=args.meta_data_dir)
