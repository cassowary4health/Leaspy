from joblib import Parallel, delayed

def leaspy_parallel_calibrate(data_iter, algo_settings_iter, leaspy_factory, leaspy_obj_cb, n_jobs=-1, **joblib_Parallel_kwargs):
    """
    Calibrate in parallel multiple Leaspy models

    Parameters
    ----------
    data_iter: an iterable of Leaspy Data objects to be calibrated on
    algo_settings_iter: an iterable of Leaspy AlgorithmSettings for every calibration task

    leaspy_factory: a function taking as input iteration index and returning a new Leaspy object that will be calibrated
    leaspy_obj_cb: a function taking as input a calibrated Leaspy object and iteration index and doing whatsoever needed with it (i.e.: saving model to a file, ...)

    n_jobs: number of parallel jobs in joblib (default to -1)
    **joblib_Parallel_kwargs: other joblib Parallel parameters (such as `verbose`, ...)

    Output
    ----------
    list of `leaspy_obj_cb` outputs for every jobs
    """

    # unitary job
    def calibrate_job(data, algo_settings, i):
        # create Leaspy object for this job from the prescribed factory
        leaspy = leaspy_factory(i)
        # calibrate model with prescribed data and settings
        leaspy.fit(data, algorithm_settings=algo_settings)
        # do something with the calibrated model
        return leaspy_obj_cb(leaspy, i)

    return Parallel(n_jobs=n_jobs, **joblib_Parallel_kwargs)(delayed(calibrate_job)(data, algo_settings, i) \
                                                             for i, (data, algo_settings) in enumerate(zip(data_iter, algo_settings_iter)))

def leaspy_parallel_personalize(leaspy_iter, data_iter, algo_settings_iter, leaspy_res_cb, n_jobs=-1, **joblib_Parallel_kwargs):
    """
    Personalize in parallel multiple Leaspy models

    Parameters
    ----------
    leaspy_iter: an iterable of Leaspy objects to personalize on
    data_iter: an iterable of Leaspy Data objects to be personalized
    algo_settings_iter: an iterable of Leaspy AlgorithmSettings for every personalization task

    leaspy_res_cb: a function taking as input a Leaspy Result object (the output of personalization task) and iteration index
                   and doing whatsoever needed with it (i.e.: saving individual parameters/results to a file, ...)

    n_jobs: number of parallel jobs in joblib (default to -1)
    **joblib_Parallel_kwargs: other joblib Parallel parameters (such as `verbose`, ...)

    Output
    ----------
    list of `leaspy_res_cb` outputs for every jobs
    """

    # unitary job
    def personalize_job(leaspy, data, algo_settings, i):
        # personalize calibrated model with prescribed data and settings
        r = leaspy.personalize(data, algo_settings)
        # do something with results of personalization
        return leaspy_res_cb(r, i)

    return Parallel(n_jobs=n_jobs, **joblib_Parallel_kwargs)(delayed(personalize_job)(leaspy, data, algo_settings, i) \
                                                             for i, (leaspy, data, algo_settings) in enumerate(zip(leaspy_iter, data_iter, algo_settings_iter)))
