from joblib import Parallel, delayed


def leaspy_parallel_calibrate(data_iter, algo_settings_iter, leaspy_factory, leaspy_obj_cb,
                              n_jobs=-1, **joblib_Parallel_kwargs):
    """
    Calibrate in parallel multiple Leaspy models.

    Parameters
    ----------
    data_iter : list [leaspy.io.data.data.Data]
        An iterable of Leaspy Data objects to be calibrated on.
    algo_settings_iter : list [leaspy.io.settings.algorithm_settings.AlgorithmSettings]
        An iterable of Leaspy AlgorithmSettings for every calibration task.
    leaspy_factory : callable
        A function taking as input iteration index and returning a new Leaspy object that will be calibrated.
    leaspy_obj_cb : callable
        A function taking as input a calibrated Leaspy object + iteration index and doing whatsoever needed with it
        (i.e.: saving model to a file, ...).
    n_jobs : int, (default -1)
        The number of parallel jobs in joblib.
    **joblib_Parallel_kwargs
        Other joblib Parallel parameters (such as `verbose`, `backend`, ...).

    Returns
    -------
    `list`
        Contains the `leaspy_obj_cb` return of each job.
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

def leaspy_parallel_update_b(data_iter, algo_settings_iter,perso_settings_iter,meta_settings_iter, leaspy_factory, leaspy_obj_cb,
                              n_jobs=-1, **joblib_Parallel_kwargs):
    """
    Calibrate in parallel multiple Leaspy models.

    Parameters
    ----------
    data_iter : list [leaspy.io.data.data.Data]
        An iterable of Leaspy Data objects to be calibrated on.
    algo_settings_iter : list [leaspy.io.settings.algorithm_settings.AlgorithmSettings]
        An iterable of Leaspy AlgorithmSettings for every calibration task.
    leaspy_factory : callable
        A function taking as input iteration index and returning a new Leaspy object that will be calibrated.
    leaspy_obj_cb : callable
        A function taking as input a calibrated Leaspy object + iteration index and doing whatsoever needed with it
        (i.e.: saving model to a file, ...).
    n_jobs : int, (default -1)
        The number of parallel jobs in joblib.
    **joblib_Parallel_kwargs
        Other joblib Parallel parameters (such as `verbose`, `backend`, ...).

    Returns
    -------
    `list`
        Contains the `leaspy_obj_cb` return of each job.
    """
    # unitary job
    @delayed
    def calibrate_job(data, algo_settings,perso_settings,meta_settings, i):
        # create Leaspy object for this job from the prescribed factory
        leaspy = leaspy_factory(i)
        # calibrate model with prescribed data and settings
        perso=leaspy.fit_B(data,algo_settings,perso_settings,meta_settings=meta_settings)
        # do something with the calibrated model
        return leaspy_obj_cb(leaspy,perso, i)

    return Parallel(n_jobs=n_jobs, **joblib_Parallel_kwargs)(
        calibrate_job(data, algo_settings,perso_settings,meta_settings, i)
        for i, (data, algo_settings,perso_settings,meta_settings)
        in enumerate(zip(data_iter, algo_settings_iter,perso_settings_iter,meta_settings_iter)))
    
