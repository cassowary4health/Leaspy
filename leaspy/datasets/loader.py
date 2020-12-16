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
    dataset_name: {'parkinson-multivariate', 'alzheimer-multivariate', 'parkinson-putamen', 'parkinson-putamen-train_and_test'}
        Name of the dataset.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame containing the IDs, timepoints and observations.

    Notes
    -----
    All `DataFrames` have the same structures.

    * Index: a `MultiIndex` - ``['ID', 'TIME']`` which contain IDs and timepoints. The `DataFrame` is sorted by index.
    So, one line corresponds to one visit for one subject. The `DataFrame` having `'tain_and_test'` in their name also
    have `'SPLIT'` as the third index level. It differenciate `train` and `test` data.

    * Columns: One column correspond to one feature (or score).
    """
    df = pd.read_csv(paths[dataset_name], dtype={'ID': str})
    if 'SPLIT' in df.columns:
        df.set_index(['ID', 'TIME', 'SPLIT'], inplace=True)
    else:
        df.set_index(['ID', 'TIME'], inplace=True)
    return df.sort_index()

# TODO: create load_model function that load models pre-trained on the available datasets
# TODO: create load_individual_parameters function that load ip computed on the available datasets
