import os

import pandas as pd

from leaspy import Leaspy, IndividualParameters

dataset_all = ['alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen',
               'parkinson-putamen-train_and_test']
model_all = ['parkinson-putamen-train']
ip_all = ['parkinson-putamen-train']

module_path = os.path.dirname(__file__)
data_paths = {dataset_name: os.path.join(module_path, 'data/' + dataset_name + '.csv') for dataset_name in dataset_all}
model_paths = {model_name: os.path.join(module_path, 'model_parameters/' + model_name + '-model_parameters.json') for model_name in model_all}
ip_paths = {ip_name: os.path.join(module_path, 'model_parameters/' + ip_name + '-individual_parameters.csv') for ip_name in ip_all}


def load_dataset(dataset_name):
    """
    Load synthetic longitudinal observations mimicking cohort of subjects with neurodegenerative disorders.

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
      So, one line corresponds to one visit for one subject. The `DataFrame` having `'train_and_test'` in their name
      also have ``'SPLIT'`` as the third index level. It differenciate `train` and `test` data.

    * Columns: One column correspond to one feature (or score).
    """
    df = pd.read_csv(data_paths[dataset_name], dtype={'ID': str})
    if 'SPLIT' in df.columns:
        df.set_index(['ID', 'TIME', 'SPLIT'], inplace=True)
    else:
        df.set_index(['ID', 'TIME'], inplace=True)
    return df.sort_index()


def load_leaspy_instance(instance_name):
    """
    Load a Leaspy instance with a model allready calibrated on the synthetic dataset corresponding to the name
    of the instance.

    Parameters
    ----------
    instance_name: {'parkinson-putamen-train'}
        Name of the instance.

    Returns
    -------
    leaspy.Leaspy
        Leaspy instance with a model allready calibrated.
    """
    return Leaspy.load(model_paths[instance_name])


def load_individual_parameters(ip_name):
    """
    Load a Leaspy instance with a model allready calibrated on the synthetic dataset corresponding to the name
    of the instance.

    Parameters
    ----------
    ip_name: {'parkinson-putamen-train'}
        Name of the individual parameters.

    Returns
    -------
    leaspy.IndividualParameters
        Leaspy instance with a model allready calibrated.
    """
    return IndividualParameters.load(model_paths[ip_name])


# TODO: add some leaspy_instance
# TODO: add some individual_parameters
