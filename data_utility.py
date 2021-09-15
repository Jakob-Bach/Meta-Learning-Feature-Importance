"""Utility for working with datasets

Functions for reading and writing data. Although the individual functions are quite short,
having a central I/O makes changes of file formats and naming schemes easier. As I/O is not a
performance bottleneck at the moment, we use plain CSV files for serialization.
"""

import pathlib
from typing import List, Optional, Union, Tuple

import pandas as pd


# Feature-part and target-part of a dataset are saved separately.
def load_dataset(dataset_name: str, directory: pathlib.Path) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]:
    X = pd.read_csv(directory / (dataset_name + '_X.csv'))
    y = pd.read_csv(directory / (dataset_name + '_y.csv'), squeeze=True)
    return X, y


def save_dataset(dataset_name: str, directory: pathlib.Path, X: Optional[pd.DataFrame] = None,
                 y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> None:
    if X is not None:
        X.to_csv(directory / (dataset_name + '_X.csv'), index=False)
    if y is not None:
        y.to_csv(directory / (dataset_name + '_y.csv'), index=False)


# List dataset names based on the files in the "directory".
def list_datasets(directory: pathlib.Path) -> List[str]:
    return [file.name.split('_X.')[0] for file in list(directory.glob('*_X.*'))]


def name_meta_target(base_model_name: str, importance_measure_name: str) -> str:
    return base_model_name + '#' + importance_measure_name
