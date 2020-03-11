from joblib import Parallel, delayed


def leaspy_parallel_calibrate(data_iter, algo_settings_iter, leaspy_factory, leaspy_obj_cb,
                              n_jobs=-1, **joblib_Parallel_kwargs):
    """
    Calibrate in parallel multiple Leaspy models.

    Parameters
    ----------
    data_iter : list [leaspy.inputs.data.data.Data]
        An iterable of Leaspy Data objects to be calibrated on.
    algo_settings_iter : list [leaspy.inputs.settings.algorithm_settings.AlgorithmSettings]
        An iterable of Leaspy AlgorithmSettings for every calibration task.
    leaspy_factory : callable
        A function taking as input iteration index and returning a new Leaspy object that will be calibrated.
    leaspy_obj_cb : callable
        A function taking as input a calibrated Leaspy object and iteration index and doing whatsoever needed with it
        (i.e.: saving model to a file, ...).
    n_jobs : int, (default -1)
        The number of parallel jobs in joblib.
    **joblib_Parallel_kwargs
        Other joblib Parallel parameters (such as `verbose`, `backend`, ...).

    Returns
    -------
    result : `list`
        Contains the `leaspy_obj_cb` outputs for every jobs.
    """
    # unitary job
    @delayed
    def calibrate_job(data, algo_settings, i):
        # create Leaspy object for this job from the prescribed factory
        leaspy = leaspy_factory(i)
        # calibrate model with prescribed data and settings
        leaspy.fit(data, algorithm_settings=algo_settings)
        # do something with the calibrated model
        return leaspy_obj_cb(leaspy, i)

    return Parallel(n_jobs=n_jobs, **joblib_Parallel_kwargs)(
        calibrate_job(data, algo_settings, i)
        for i, (data, algo_settings)
        in enumerate(zip(data_iter, algo_settings_iter))
    )


def leaspy_parallel_personalize(leaspy_iter, data_iter, algo_settings_iter, leaspy_res_cb,
                                n_jobs=-1, **joblib_Parallel_kwargs):
    """
    Personalize in parallel multiple Leaspy models

    Parameters
    ----------
    leaspy_iter : list [leaspy.Leaspy]
        An iterable of Leaspy objects to personalize on
    data_iter : list [leaspy.inputs.data.data.Data]
        An iterable of Leaspy Data objects to be calibrated on.
    algo_settings_iter : list [leaspy.inputs.settings.algorithm_settings.AlgorithmSettings]
        An iterable of Leaspy AlgorithmSettings for every calibration task.
    leaspy_res_cb : callable
        A function taking as input a Leaspy Result object (the output of personalization task) and iteration index
        and doing whatsoever needed with it (i.e.: saving individual parameters/results to a file, ...).
    n_jobs : int, (default -1)
        The number of parallel jobs in joblib.
    **joblib_Parallel_kwargs
        Other joblib Parallel parameters (such as `verbose`, `backend`, ...).

    Returns
    -------
    `list`
        Contains the `leaspy_res_cb` outputs for every jobs.
    """
    # unitary job
    @delayed
    def personalize_job(leaspy, data, algo_settings, i):
        # personalize calibrated model with prescribed data and settings
        r = leaspy.personalize(data, algo_settings)
        # do something with results of personalization
        return leaspy_res_cb(r, i)

    return Parallel(n_jobs=n_jobs, **joblib_Parallel_kwargs)(
        personalize_job(leaspy, data, algo_settings, i)
        for i, (leaspy, data, algo_settings)
        in enumerate(zip(leaspy_iter, data_iter, algo_settings_iter))
    )
