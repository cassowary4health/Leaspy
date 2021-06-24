import os
import json
import warnings

from leaspy import Leaspy, AlgorithmSettings, Data


from .parallel import leaspy_parallel_personalize



def stick_to_features_with_warns(df, expected_fts):
    """
    Check for extra/missing features compared to expected_fts, and warns if any
    Return df with exactly expected features (dropped/padded with nan in either case to be compliant, also reordering in expected order)
    """
    extra_fts = df.columns.difference(expected_fts).tolist()
    if len(extra_fts) > 0:
        warnings.warn(f'Some extra features are present: {extra_fts}, skipping them.')

    missing_fts = list(set(expected_fts).difference(df.columns))
    if len(missing_fts) > 0:
        warnings.warn(f'Some features are missing: {missing_fts}, padding them (nan).')

    return df.reindex(columns=expected_fts)

def personalize_resampling(df,
                           path_calibrate,
                           path_output_personalize,
                           algo_settings_personalize, leaspy_ips_cb, *,
                           n_jobs = -1, parallel_kwargs = {},
                           indices_skipper = lambda _: []):
    """
    Parameters
    ----------
    df: pandas.DataFrame
        data to personalize our models on

    path_calibrate: string
        root folder where calibration results were saved

    path_output_personalize: string
        root folder to save personalize results in

    algo_settings_personalize: leaspy.AlgortihmSettings
        personalization algo settings
        (same for all runs, set seed on them to be reproducible)

    leaspy_ips_cb: function leaspy.IndividualParameters, iter_index -> *
        cf. `leaspype.functions.parallel.leaspy_parallel_personalize`

    n_jobs: int (default -1)
        number of jobs, as expected by joblib Parallel

    parallel_kwargs: dict (default {})
        extra kwargs passed to joblib.Parallel beyond n_jobs

    indices_skipper: function: list[calibrated patiens ID for this run] -> list[indices of df to skip in personalization for this run]
        e.g. lambda lp: lp (skip patients who were used in calibration, sometimes useful if personalizing on same cohort than calibration)
        e.g. lambda lp: [] (skip none, default)

    setup_api: str (default `leaspype.__api__`)
        version/api info to handle long-term compatibility between experimental setups

    Returns
    -------
    List of all outputs from leaspy_ips_cb
    """

    #%% Various checks

    assert isinstance(algo_settings_personalize, AlgorithmSettings)

  

    # load resampling indices: dict(fold: [list_train_ids, list_test_ids])
    with open(os.path.join(path_calibrate, "resampling_indices.json"), "r") as json_file:
        calib_indices = json.load(json_file)

    # Data as list
    data_iter = []
    for cv_iter, fold_indices in calib_indices.items():
        skip_indices = indices_skipper(fold_indices[0]) # pass list of patients IDs (from calibration cohort) used during calibration
        df_split = df.drop(skip_indices)
        #df_split = df.loc[include_indices] # TODO : choose inclusion method
        data = Data.from_dataframe(df_split.reset_index())
        data_iter.append(data)

    # TODO: Save who we personalized on?
    # Looking at indices of individual_parameters should be enough...

    # Get calibrated models paths
    model_paths = [os.path.join(path_calibrate, f"fold_{fold_iter}", "model_parameters.json") for fold_iter in calib_indices.keys()]

    # Load estimated leaspy models
    # will fail if one model is missing (calibration not finished)
    leaspy_iter = [Leaspy.load(mp) for mp in model_paths]

    # Algo settings iter
    algo_settings_personalize_iter = [algo_settings_personalize for _ in leaspy_iter]

    # Save algo settings (same for all runs)
    algo_settings_personalize.save(os.path.join(path_output_personalize, "algo_settings_personalize.json"), indent=2)

    return leaspy_parallel_personalize(leaspy_iter, data_iter, algo_settings_personalize_iter, leaspy_ips_cb,
                                       n_jobs=n_jobs, **parallel_kwargs)

def personalize_resampling_kernel(df,
                           path_calibrate,
                           path_output_personalize,
                           algo_settings_personalize, leaspy_ips_cb, *,n_comp=0,
                           n_jobs = -1, parallel_kwargs = {},
                           indices_skipper = lambda _: []):
    """
    Parameters
    ----------
    df: pandas.DataFrame
        data to personalize our models on

    path_calibrate: string
        root folder where calibration results were saved

    path_output_personalize: string
        root folder to save personalize results in

    algo_settings_personalize: leaspy.AlgortihmSettings
        personalization algo settings
        (same for all runs, set seed on them to be reproducible)

    leaspy_ips_cb: function leaspy.IndividualParameters, iter_index -> *
        cf. `leaspype.functions.parallel.leaspy_parallel_personalize`

    n_jobs: int (default -1)
        number of jobs, as expected by joblib Parallel

    parallel_kwargs: dict (default {})
        extra kwargs passed to joblib.Parallel beyond n_jobs

    indices_skipper: function: list[calibrated patiens ID for this run] -> list[indices of df to skip in personalization for this run]
        e.g. lambda lp: lp (skip patients who were used in calibration, sometimes useful if personalizing on same cohort than calibration)
        e.g. lambda lp: [] (skip none, default)

    setup_api: str (default `leaspype.__api__`)
        version/api info to handle long-term compatibility between experimental setups

    Returns
    -------
    List of all outputs from leaspy_ips_cb
    """

    #%% Various checks

    assert isinstance(algo_settings_personalize, AlgorithmSettings)

    # check that a correct calibration setup was given and load it
  


    # load resampling indices: dict(fold: [list_train_ids, list_test_ids])
    with open(os.path.join(path_calibrate, "resampling_indices.json"), "r") as json_file:
        calib_indices = json.load(json_file)

    # Data as list
    data_iter = []
    for cv_iter, fold_indices in calib_indices.items():
        skip_indices = indices_skipper(fold_indices[0]) # pass list of patients IDs (from calibration cohort) used during calibration
        df_split = df.drop(skip_indices)
        #df_split = df.loc[include_indices] # TODO : choose inclusion method
        data = Data.from_dataframe(df_split.reset_index())
        data_iter.append(data)

    # TODO: Save who we personalized on?
    # Looking at indices of individual_parameters should be enough...

    # Get calibrated models paths
    model_paths = [os.path.join(path_calibrate, f"fold_{fold_iter}", "model_parameters.json") for fold_iter in calib_indices.keys()]

    # Load estimated leaspy models
    # will fail if one model is missing (calibration not finished)
    def Loadcomp(path,n_comp):
        Mod=Leaspy.load(path)
        L=Mod.model.saveB[:n_comp]
        D=Mod.model.saveParam[n_comp]
        Mod.model.load_parameters(D)
        Mod.model.saveB=L
        Mod.model.initBlink()
        Mod.model.reconstructionB()
        return Mod
    leaspy_iter = [Loadcomp(mp,n_comp) for mp in model_paths]

    # Algo settings iter
    algo_settings_personalize_iter = [algo_settings_personalize for _ in leaspy_iter]

    # Save algo settings (same for all runs)
    algo_settings_personalize.save(os.path.join(path_output_personalize, "algo_settings_personalize.json"), indent=2)

    return leaspy_parallel_personalize(leaspy_iter, data_iter, algo_settings_personalize_iter, leaspy_ips_cb,
                                       n_jobs=n_jobs, **parallel_kwargs)
