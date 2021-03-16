from joblib import Parallel, delayed
import warnings

def leaspy_parallel_calibrate(data_iter, algo_settings_iter, leaspy_factory, leaspy_obj_cb,
                              n_jobs=-1, **joblib_Parallel_kwargs):
    """
    .. deprecated:: 1.0
    Calibrate in parallel multiple Leaspy models.

    Parameters
    ----------
    data_iter : list [ :class:`.Data` ]
        An iterable of Leaspy Data objects to be calibrated on.
    algo_settings_iter : list [ :class:`.AlgorithmSettings` ]
        An iterable of Leaspy AlgorithmSettings for every calibration task.
    leaspy_factory : callable
        A function taking as input iteration index and returning a new :class:`.Leaspy` object that will be calibrated.
    leaspy_obj_cb : callable
        A function taking as input a calibrated :class:`.Leaspy` object and iteration index and doing whatsoever needed with it
        (i.e.: saving model to a file, ...).
    n_jobs : int (default -1)
        The number of parallel jobs in :mod:`joblib`.
    **joblib_Parallel_kwargs
        Other :class:`joblib.Parallel` parameters (such as `verbose`, `backend`, ...).

    Returns
    -------
    list
        Contains the `leaspy_obj_cb` return of each job.
    """
    warnings.warn('leaspy_parallel_calibrate function is deprecated. Please use the one in Leaspype', DeprecationWarning)

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


def leaspy_parallel_personalize(leaspy_iter, data_iter, algo_settings_iter, leaspy_ips_cb,
                                n_jobs=-1, **joblib_Parallel_kwargs):
    """
    .. deprecated:: 1.0
    Personalize in parallel multiple Leaspy models

    Parameters
    ----------
    leaspy_iter : list [ :class:`.Leaspy` ]
        An iterable of Leaspy objects to personalize on
    data_iter : list [ :class:`.Data` ]
        An iterable of Leaspy Data objects to be calibrated on.
    algo_settings_iter : list [ :class:`.AlgorithmSettings` ]
        An iterable of Leaspy AlgorithmSettings for every calibration task.
    leaspy_ips_cb : callable
        A function taking as input :
            * the output of personalization task: :class:`.IndividualParameters`
            * the iteration index: uint
        and doing whatsoever needed with it (e.g.: saving individual parameters to a file, ...).
    n_jobs : int (default -1)
        The number of parallel jobs in :mod:`joblib`.
    **joblib_Parallel_kwargs
        Other :class:`joblib.Parallel` parameters (such as `verbose`, `backend`, ...).

    Returns
    -------
    list
        Contains the `leaspy_ips_cb` return of each job.
    """
    warnings.warn('leaspy_parallel_personalize function is deprecated. Please use the one in Leaspype', DeprecationWarning)
    # unitary job

    @delayed
    def personalize_job(leaspy, data, algo_settings, i):
        # personalize calibrated model with prescribed data and settings
        ips = leaspy.personalize(data, algo_settings)
        # do something with results of personalization
        return leaspy_ips_cb(ips, i)

    return Parallel(n_jobs=n_jobs, **joblib_Parallel_kwargs)(
        personalize_job(leaspy, data, algo_settings, i)
        for i, (leaspy, data, algo_settings)
        in enumerate(zip(leaspy_iter, data_iter, algo_settings_iter))
    )
