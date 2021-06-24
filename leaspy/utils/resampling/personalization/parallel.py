from joblib import Parallel, delayed


def leaspy_parallel_personalize(leaspy_iter, data_iter, algo_settings_iter, leaspy_ips_cb,
                                n_jobs=-1, **joblib_Parallel_kwargs):
    """
    Personalize in parallel multiple Leaspy models

    Parameters
    ----------
    leaspy_iter : list [leaspy.Leaspy]
        An iterable of Leaspy objects to personalize on
    data_iter : list [leaspy.io.data.data.Data]
        An iterable of Leaspy Data objects to be calibrated on.
    algo_settings_iter : list [leaspy.io.settings.algorithm_settings.AlgorithmSettings]
        An iterable of Leaspy AlgorithmSettings for every calibration task.
    leaspy_ips_cb : callable
        A function taking as input :
        - the output of personalization task: leaspy.io.outputs.individual_parameters.IndividualParameters
        - the iteration index: uint
        and doing whatsoever needed with it (e.g.: saving individual parameters to a file, ...).
    n_jobs : int, (default -1)
        The number of parallel jobs in joblib.
    **joblib_Parallel_kwargs
        Other joblib Parallel parameters (such as `verbose`, `backend`, ...).

    Returns
    -------
    `list`
        Contains the `leaspy_ips_cb` return of each job.
    """
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
