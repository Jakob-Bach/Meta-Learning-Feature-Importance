"""Utility for working with datasets

Functions for reading and writing data. Although the individual functions are quite short,
having a central I/O makes changes of file formats and naming schemes easier. As I/O is not a
performance bottleneck at the moment, we use plain CSV files for serialization.
"""

import pathlib
from typing import Union, Tuple

import pandas as pd


# Feature-part and target-part of a dataset are saved separately.
def load_dataset(dataset_name: str, directory: pathlib.Path) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]:
    X = pd.read_csv(directory / (dataset_name + '_X.csv'))
    y = pd.read_csv(directory / (dataset_name + '_y.csv'), squeeze=True)
    return X, y


def save_dataset(X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], dataset_name: str,
                 directory: pathlib.Path) -> None:
    X.to_csv(directory / (dataset_name + '_X.csv'), index=False)
    y.to_csv(directory / (dataset_name + '_y.csv'), index=False)