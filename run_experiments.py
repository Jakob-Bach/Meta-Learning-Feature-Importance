"""Run experiments

Script for the experimental pipeline. Determines and stores prediction performances for all
combinations of
- base datasets and their corresponding meta-features
- feature-importance measures (part of meta-target)
- base models (part of meta-target)
- meta-models

Should be run after the dataset-preparation script.

Usage: python -m run_experiments --help
"""

import argparse
import multiprocessing
import pathlib
from typing import Optional

import pandas as pd
import tqdm

import data_utility
import meta_models
import meta_targets


# Evaluate all meta-models for one meta-target (whose position is denoted by "target_column_idx")
# of a meta-dataset stored in "meta_data_dir", the file name containing "dataset_name".
def evaluate_meta_models(dataset_name: str, meta_data_dir: pathlib.Path,
                         target_column_idx: int) -> pd.DataFrame:
    prediction_results = []
    X, y = data_utility.load_dataset(dataset_name=dataset_name, directory=meta_data_dir)
    target_name = y.columns[target_column_idx]
    y = y[target_name]
    for model_name, model_dict in meta_models.META_MODELS.items():
        prediction_result = meta_models.predict_and_evaluate(
            X=X, y=y, model_func=model_dict['func'], model_args=model_dict['args'])
        prediction_result['meta_model'] = model_name
    prediction_results = pd.concat(prediction_results)
    prediction_results['dataset'] = dataset_name
    prediction_results['meta_target'] = target_name
    return prediction_results


# Evaluate all meta-models for all meta-datasets in "meta_data_dir". Save the results in
# "results_dir".
def run_meta_prediction_experiments(meta_data_dir: pathlib.Path, results_dir: pathlib.Path,
                                    n_processes: Optional[int] = None) -> None:
    settings_list = [
        {'dataset_name': dataset_name, 'meta_data_dir': meta_data_dir, 'target_column_idx': i}
        for dataset_name in data_utility.list_datasets(directory=meta_data_dir)
        for i in range(len(meta_targets.IMPORTANCE_MEASURES) * len(meta_targets.BASE_MODELS))
    ]  # assumes same number of importance measures and base models is used for each base dataset
    with tqdm.tqdm(total=len(settings_list), desc='Evaluating meta-models') as progress_bar:
        with multiprocessing.Pool(processes=n_processes) as process_pool:
            results = [process_pool.apply_async(evaluate_meta_models, kwds=settings,
                                                callback=lambda x: progress_bar.update())
                       for settings in settings_list]
            results = pd.concat([x.get() for x in results])
            results.to_csv(results_dir / 'meta_prediction_results.csv', index=False)


# Use MetaLFI as feature-selection approach and compare against other feature-selection approaches.
def run_feature_selection_experiments(
        base_data_dir: pathlib.Path, meta_data_dir: pathlib.Path, results_dir: pathlib.Path,
        n_processes: Optional[int] = None) -> None:
    pass


# Do some sanity checks regarding the paths and then run the experiments.
def run_experiments(base_data_dir: pathlib.Path, meta_data_dir: pathlib.Path,
                    results_dir: pathlib.Path, n_processes: Optional[int] = None) -> None:
    print('Experimental pipeline started.')
    if not base_data_dir.is_dir():
        raise FileNotFoundError('Base-dataset directory does not exist.')
    if not meta_data_dir.is_dir():
        raise FileNotFoundError('Meta-dataset directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    run_meta_prediction_experiments(meta_data_dir=meta_data_dir, results_dir=results_dir,
                                    n_processes=n_processes)
    run_feature_selection_experiments(base_data_dir=base_data_dir, meta_data_dir=meta_data_dir,
                                      results_dir=results_dir, n_processes=n_processes)
    print('Experimental pipeline executed successfully.')


# Parse command-line arguments and run pipeline.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs the experimental pipeline. Might take a while.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--base_data_dir', type=pathlib.Path, default='data/base_datasets/',
                        help='Directory with base datasets (inputs to pipleline).')
    parser.add_argument('-m', '--meta_data_dir', type=pathlib.Path, default='data/meta_datasets/',
                        help='Directory with meta-datasets (inputs to pipleline).')
    parser.add_argument('-r', '--results_dir', type=pathlib.Path, default='data/results/',
                        help='Directory for experimental results (outputs of pipeline). ' +
                        'Will be created if necessary.')
    parser.add_argument('-p', '--n_processes', type=int, default=None,
                        help='Number of processes for multi-processing (default: all cores).')
    run_experiments(**vars(parser.parse_args()))
