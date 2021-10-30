# Meta-Learning Feature Importance

This repository contains the code of the planned paper

> Bach, Jakob, and Cem Özcan. "Meta-Learning Feature Importance"

This is a re-implementation of [MetaLFI](https://github.com/CemOezcan/metalfi).
The paper won't appear though, as this research project was discontinued 🙃.
This document should describe the steps to reproduce the experiments.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all necessary dependencies.
Our code is implemented in Python (version 3.8; other versions, including lower ones, might work as well).

### Option 1: `conda` Environment

If you use `conda`, you can install the right Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.8
conda activate <conda-env-name>
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.7; other versions might work as well) to create an environment for our experiments.
First, make sure you have the right Python version available.
Next, you can install `virtualenv` with

```bash
python -m pip install virtualenv==20.4.7
```

To set up an environment with `virtualenv`, run

```bash
python -m virtualenv -p <path/to/right/python/executable> <path/to/env/destination>
```

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, simply run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

To leave the environment, run

```bash
deactivate
```

### Optional Dependencies

To use the environment in the IDE `Spyder`, you need to install `spyder-kernels` into the environment.

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
First, run

```bash
python -m prepare_datasets
```

to download the base datasets and prepare the meta-datasets, i.e., compute meta-features and meta-targets.
Next, start the experimental pipeline with

```bash
python -m run_experiments
```

To print statistics and create the plots for the paper, run

```bash
python -m run_evaluation
```

(This script does not exist yet.)

All scripts have a few command line options, which you can see by running the scripts like

```bash
python -m run_experiments --help
```
