
import os
import json
import copy

import numpy as np

from leaspy import Leaspy, AlgorithmSettings, Data


from .parallel import leaspy_parallel_calibrate
from .parallel import leaspy_parallel_update_b



def calibrate_resampling(df, leaspy_factory, algo_settings, patients_splitter, path_output, *,
                         n_jobs = -1, parallel_kwargs = {}, logs_kwargs = {}):
    """
    Parameters
    ----------
    df: pandas.DataFrame
        must have [ID,TIME] + columns used as features (leaspy-ready)

    leaspy_factory: function: iter_index -> leaspy.Leaspy object to calibrate
        cf. `leaspype.functions.parallel.leaspy_parallel_calibrate`

    algo_settings: leaspy.AlgorithmSettings
        calibration algo settings
        (same for all runs, set seed on them to be reproducible)

    patients_splitter: object supporting sklearn "split" interface
        object to resample patients between runs.
        e.g. sklearn.model_selection.RepeteadKFold(n_splits=5, n_repeats=20, random_state=0)
        If it is None, than leaspy model will be calibrated on the whole dataset.

    output_path: string
        root directory to store resulting files

    n_jobs: int (default -1)
        number of jobs, as expected by joblib Parallel

    parallel_kwargs: dict (default {})
        extra kwargs passed to joblib.Parallel beyond n_jobs

    logs_kwargs: dict (default {}) or None (no logs)
        if a dict, cf. kwargs of AlgorithmSettings.set_logs,
        keys in ['console_print_periodicity', 'plot_periodicity', 'save_periodicity']


    Returns
    -------
    List of all Leaspy calibrated models
    """

    # Check input is ok

    leaspy_sample_obj = leaspy_factory(-1)
    assert isinstance(leaspy_sample_obj, Leaspy)
    assert isinstance(algo_settings, AlgorithmSettings)
    if patients_splitter:
        assert hasattr(patients_splitter, 'split') # TODO: be more generic? just a list of splits?

    

    # Create the resampling procedure and get indices in train/test for each run
    patients = np.array(df.index.unique('ID'))

    # <!> no run with a calibration on whole dataframe
    resampling_indices = {}
    if patients_splitter:
        for j, (train_index, test_index) in enumerate(patients_splitter.split(patients)):
            patients_train, patients_test = patients[train_index], patients[test_index]
            resampling_indices[j] = (patients_train.tolist(), patients_test.tolist())  # 0: calibrated on, 1: ignored
    else:
        resampling_indices[0] = (patients.tolist(), [])

    # Save experiment infos
    df.to_csv(os.path.join(path_output, "df.csv"))
    with open(os.path.join(path_output, "resampling_indices.json"), "x") as json_file:
        json.dump(resampling_indices, json_file, indent=2)

    algo_settings.save(os.path.join(path_output, "algo_settings.json"), indent=2)

    fold_subdir = lambda fold_iter: os.path.join(path_output, f"fold_{fold_iter}")

    # Run Leaspy

    # Data as list
    data_iter = []
    for fold_iter, fold_indices in resampling_indices.items():
        df_split = df.loc[fold_indices[0]]
        data = Data.from_dataframe(df_split.reset_index())
        data_iter.append(data)

        # create subdir
        os.makedirs(fold_subdir(fold_iter), exist_ok=False) # no rewrite

    ## Algo settings as list (same for all calibs but <!> separate logs)

    if logs_kwargs is None: # no logs
        algo_settings_iter = [algo_settings for _ in data_iter]
    else:
        algo_settings_iter = []
        for fold_iter in resampling_indices.keys():
            fold_logs_path = os.path.join(fold_subdir(fold_iter), 'logs')
            os.makedirs(fold_logs_path, exist_ok=False) # no rewrite
            fold_algo_settings = copy.deepcopy(algo_settings)
            fold_algo_settings.set_logs(fold_logs_path, **logs_kwargs)
            algo_settings_iter.append(fold_algo_settings)

    def leaspy_callback(leaspy_model, fold_iter):
        leaspy_model.save(os.path.join(fold_subdir(fold_iter), 'model_parameters.json'), indent=2)
        return leaspy_model

    # Calibrate all the models with resampled data
    return leaspy_parallel_calibrate(data_iter, algo_settings_iter, leaspy_factory, leaspy_callback,
                                     n_jobs=n_jobs, **parallel_kwargs)


def update_b_resampling(df, leaspy_factory, algo_settings,perso_settings,meta_settings, patients_splitter, path_output, *,
                         n_jobs = -1, parallel_kwargs = {}, logs_kwargs = {}):
    """
    Parameters changer
    ----------
    df: pandas.DataFrame
        must have [ID,TIME] + columns used as features (leaspy-ready)

    leaspy_factory: function: iter_index -> leaspy.Leaspy object to calibrate
        cf. `leaspype.functions.parallel.leaspy_parallel_calibrate`

    algo_settings: leaspy.AlgorithmSettings
        calibration algo settings
        (same for all runs, set seed on them to be reproducible)

    patients_splitter: object supporting sklearn "split" interface
        object to resample patients between runs.
        e.g. sklearn.model_selection.RepeteadKFold(n_splits=5, n_repeats=20, random_state=0)
        If it is None, than leaspy model will be calibrated on the whole dataset.

    output_path: string
        root directory to store resulting files

    n_jobs: int (default -1)
        number of jobs, as expected by joblib Parallel

    parallel_kwargs: dict (default {})
        extra kwargs passed to joblib.Parallel beyond n_jobs

    logs_kwargs: dict (default {}) or None (no logs)
        if a dict, cf. kwargs of AlgorithmSettings.set_logs,
        keys in ['console_print_periodicity', 'plot_periodicity', 'save_periodicity']



    Returns
    -------
    List of all Leaspy calibrated models
    """

    # Check input is ok changer perso setttings meta settings gérer les enregistrements, id parameters + paramaeters

    leaspy_sample_obj = leaspy_factory(-1)
    assert isinstance(leaspy_sample_obj, Leaspy)
    assert isinstance(algo_settings, AlgorithmSettings)
    if patients_splitter:
        assert hasattr(patients_splitter, 'split') # TODO: be more generic? just a list of splits?

   

    # Create the resampling procedure and get indices in train/test for each run
    patients = np.array(df.index.unique('ID'))

    # <!> no run with a calibration on whole dataframe
    resampling_indices = {}
    if patients_splitter:
        for j, (train_index, test_index) in enumerate(patients_splitter.split(patients)):
            patients_train, patients_test = patients[train_index], patients[test_index]
            resampling_indices[j] = (patients_train.tolist(), patients_test.tolist())  # 0: calibrated on, 1: ignored
    else:
        resampling_indices[0] = (patients.tolist(), [])

    # Save experiment infos
    df.to_csv(os.path.join(path_output, "df.csv"))
    with open(os.path.join(path_output, "resampling_indices.json"), "x") as json_file:
        json.dump(resampling_indices, json_file, indent=2)

    algo_settings.save(os.path.join(path_output, "algo_settings.json"), indent=2)
    perso_settings.save(os.path.join(path_output, "perso_settings.json"), indent=2)
    with open(os.path.join(os.path.join(path_output, "meta_settings.json")), "w") as json_file:
        json.dump(meta_settings, json_file)
   
    fold_subdir = lambda fold_iter: os.path.join(path_output, f"fold_{fold_iter}")

    # Run Leaspy

    # Data as list
    data_iter = []
    for fold_iter, fold_indices in resampling_indices.items():
        df_split = df.loc[fold_indices[0]]
        data = Data.from_dataframe(df_split.reset_index())
        data_iter.append(data)

        # create subdir
        os.makedirs(fold_subdir(fold_iter), exist_ok=False) # no rewrite

    ## Algo settings as list (same for all calibs but <!> separate logs)

    if logs_kwargs is None: # no logs
        algo_settings_iter = [algo_settings for _ in data_iter]
    else:
        algo_settings_iter = []
        for fold_iter in resampling_indices.keys():
            fold_logs_path = os.path.join(fold_subdir(fold_iter), 'logs')
            os.makedirs(fold_logs_path, exist_ok=False) # no rewrite
            fold_algo_settings = copy.deepcopy(algo_settings)
            fold_algo_settings.set_logs(fold_logs_path, **logs_kwargs)
            algo_settings_iter.append(fold_algo_settings)
    perso_settings_iter = [perso_settings for _ in data_iter]
    meta_settings_iter = [meta_settings for _ in data_iter]
    

    def leaspy_callback(leaspy_model,perso, fold_iter):
        leaspy_model.save(os.path.join(fold_subdir(fold_iter), 'model_parameters.json'), indent=2)
        perso.save(os.path.join(fold_subdir(fold_iter), 'individual_parameters.json'), indent=2)
        return leaspy_model

    # Calibrate all the models with resampled data
    return leaspy_parallel_update_b(data_iter, algo_settings_iter,perso_settings_iter,meta_settings_iter, leaspy_factory, leaspy_callback,
                                     n_jobs=n_jobs, **parallel_kwargs)
