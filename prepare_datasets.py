"""Prepare datasets

Script that:
- downloads, pre-processes, and saves base datasets from OpenML
- computes meta-features
- computes meta-targets (combining feature-importance measures and base models)
- saves the meta-datasets

Usage: python -m prepare_datasets --help
"""

import argparse
import multiprocessing
import pathlib
from typing import Collection, Dict, Optional, Sequence, Union
import warnings

import numpy as np
import openml
import pandas as pd
import sklearn.preprocessing
import tqdm

import data_utility
import meta_features
import meta_targets


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
    data_utility.save_dataset(X=X, y=y, dataset_name=dataset.name, directory=base_data_dir)


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


# Compute all meta-features for one base dataset with "base_dataset_name", located in
# "base_data_dir", and store the resulting meta-data in "meta_data_dir"
def compute_and_save_meta_features(base_data_dir: pathlib.Path, base_dataset_name: str,
                                   meta_data_dir: pathlib.Path) -> None:
    X, y = data_utility.load_dataset(dataset_name=base_dataset_name, directory=base_data_dir)
    result = meta_features.compute_meta_features(X=X, y=y)
    data_utility.save_dataset(dataset_name=base_dataset_name, directory=meta_data_dir, X=result)


# For each base dataset from "base_data_dir", compute all meta-features. Save the resulting
# meta-data into "meta_data_dir".
def prepare_meta_features(base_data_dir: pathlib.Path, meta_data_dir: pathlib.Path,
                          n_processes: Optional[int] = None) -> None:
    print('Meta-feature preparation started.')
    base_datasets = data_utility.list_datasets(directory=base_data_dir)
    with tqdm.tqdm(total=(len(base_datasets)), desc='Computing meta-features') as progress_bar:
        with multiprocessing.Pool(processes=n_processes) as process_pool:
            results = [process_pool.apply_async(compute_and_save_meta_features, kwds={
                'base_data_dir': base_data_dir, 'base_dataset_name': base_dataset_name,
                'meta_data_dir': meta_data_dir}, callback=lambda x: progress_bar.update())
                for base_dataset_name in base_datasets]
            [x.wait() for x in results]  # don't need to return value here, just wait till finished
    print('Meta-features prepared and saved.')


# Compute one meta-target, i.e., apply one importance measure and one base model to one base
# dataset. Return the actual meta-target (numeric feature importances) and some information
# identifying it.
def compute_meta_target(base_data_dir: pathlib.Path, base_dataset_name: str, base_model_name: str,
                        importance_measure_name: str) -> Dict[str, Union[str, Collection[float]]]:
    result = {'base_dataset': base_dataset_name, 'base_model': base_model_name,
              'importance_measure': importance_measure_name}
    X, y = data_utility.load_dataset(dataset_name=base_dataset_name, directory=base_data_dir)
    importance_type = meta_targets.IMPORTANCE_MEASURES[importance_measure_name]
    base_model_func = meta_targets.BASE_MODELS[base_model_name]['func']
    base_model_args = meta_targets.BASE_MODELS[base_model_name]['args']
    result['values'] = importance_type.compute_importance(X=X, y=y, model_func=base_model_func,
                                                          model_args=base_model_args)
    return result


# For each base dataset from "base_data_dir", compute all meta-targets, i.e., all
# feature-importance measures for all base models. Save the resulting meta-data into
# "meta_data_dir".
def prepare_meta_targets(base_data_dir: pathlib.Path, meta_data_dir: pathlib.Path,
                         n_processes: Optional[int] = None) -> None:
    print('Meta-target preparation started.')
    base_datasets = data_utility.list_datasets(directory=base_data_dir)
    with tqdm.tqdm(total=(len(base_datasets) * len(meta_targets.IMPORTANCE_MEASURES) *
                          len(meta_targets.BASE_MODELS)), desc='Computing meta-targets') as progress_bar:
        with multiprocessing.Pool(processes=n_processes) as process_pool:
            results = [process_pool.apply_async(compute_meta_target, kwds={
                'base_data_dir': base_data_dir, 'base_dataset_name': base_dataset_name,
                'base_model_name': base_model_name, 'importance_measure_name': importance_measure_name
                }, callback=lambda x: progress_bar.update())
                for base_dataset_name in base_datasets
                for base_model_name in meta_targets.BASE_MODELS.keys()
                for importance_measure_name in meta_targets.IMPORTANCE_MEASURES.keys()]
            results = [x.get() for x in results]
    # Combine individual meta-targets to one data frame per base dataset:
    meta_target_data = {base_dataset_name: pd.DataFrame() for base_dataset_name in base_datasets}
    for result in results:
        column_name = data_utility.name_meta_target(
            importance_measure_name=result['importance_measure'],
            base_model_name=result['base_model'])
        meta_target_data[result['base_dataset']][column_name] = result['values']
    for base_dataset_name, data_frame in meta_target_data.items():
        data_utility.save_dataset(dataset_name=base_dataset_name, directory=meta_data_dir,
                                  y=data_frame)
    print('Meta-targets prepared and saved.')


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
    parser.add_argument('-p', '--n_processes', type=int, default=None,
                        help='Number of processes for multi-processing (default: all cores).')
    args = parser.parse_args()
    prepare_base_datasets(base_data_dir=args.base_data_dir, data_ids=args.data_ids)
    prepare_meta_features(base_data_dir=args.base_data_dir, meta_data_dir=args.meta_data_dir,
                          n_processes=args.n_processes)
    prepare_meta_targets(base_data_dir=args.base_data_dir, meta_data_dir=args.meta_data_dir,
                         n_processes=args.n_processes)
