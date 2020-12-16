import os

import pandas as pd

dataset_all = ['alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen',
               'parkinson-putamen-train_and_test']
module_path = os.path.dirname(__file__)
paths = {dataset_name: os.path.join(module_path, 'data/' + dataset_name + '.csv') for dataset_name in dataset_all}


def load_dataset(dataset_name):
    """
    Load synthetic & longitudinal neurodegenerative markers observations of virtual subjects.

    Parameters
    ----------
    dataset_name: {'parkinson-multivariate', 'alzheimer-multivariate'}
        Name of the dataset.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(paths[dataset_name], dtype={'ID': str})
    if 'SPLIT' in df.columns:
        df.set_index(['ID', 'TIME', 'SPLIT'], inplace=True)
    else:
        df.set_index(['ID', 'TIME'], inplace=True)
    return df

# TODO: create load_model function that load models pre-trained on the available datasets
# TODO: create load_individual_parameters function that load ip computed on the available datasets
