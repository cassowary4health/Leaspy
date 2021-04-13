import numpy as np
import warnings

from leaspy import IndividualParameters


def append_spaceshifts_to_individual_parameters_dataframe(df_individual_parameters, leaspy):
    r"""
    .. deprecated:: 1.0
    Returns a new dataframe with space shift columns

    Parameters
    ----------
    df_individual_parameters: :class:`pandas.DataFrame`
        Dataframe of the individual parameters. Each row corresponds to an individual. The index is the index of the patient.
    leaspy: Leaspy
        Initialize model

    Returns
    -------
    :class:`pandas.DataFrame`
        Copy of the initial dataframe with additional columns being the space shifts of the individuals.

    """
    warnings.warn('append_spaceshifts_to_individual_parameters_dataframe function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    df_ip = df_individual_parameters.copy()

    sources = df_ip [['sources_' + str(i) for i in range(leaspy.model.source_dimension)]].values.T
    spaceshifts = np.dot(leaspy.model.attributes.mixing_matrix, sources)

    for i, spaceshift_coord in enumerate(spaceshifts):
        df_ip['w_' + str(i)] = spaceshift_coord

    return df_ip


def get_reparametrized_ages(ages, individual_parameters, leaspy):
    r"""
    .. deprecated:: 1.0
    Reparametrize the real ages of the patients onto the pathological timeline

    Parameters
    ----------
    individual_parameters: Individual parameters object
        Contains the individual parameters for each patient

    ages: dict {patient_idx: [ages]}
        Contains the patient ages to reparametrized

    leaspy: Leaspy object
        Contains the model parameters

    Returns
    -------
    reparametrized_ages: dict {patient_idx: [reparametrized_ages]}
        Contains the reparametrized ages

    Raises
    ------
    ValueError:
        If one of the index not in the individual parameters

    Examples
    --------

    >>> ages = {'idx-1': [78, 79, 81], 'idx-2': [67, 68, 74], 'idx-3': [56]}
    >>> repametrized_ages = get_reparametrized_ages(ages, individual_parameters, leaspy)
    """

    warnings.warn('get_reparametrized_ages function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    tau_mean = leaspy.model.parameters['tau_mean']
    indices = individual_parameters._indices
    reparametrized_ages = {}

    for idx, ages in ages.items():
        if idx not in indices:
            raise ValueError(f'The index {idx} is not in the individual parameters')

        idx_ip = individual_parameters[idx]
        alpha = np.exp(idx_ip['xi'])
        tau = idx_ip['tau']

        reparam_ages = [alpha * (age - tau ) + tau_mean for age in ages]
        reparametrized_ages[idx] = [_.numpy().tolist() for _ in reparam_ages]

    return reparametrized_ages


def compute_trajectory_of_population(timepoints, individual_parameters, leaspy):
    r"""
    .. deprecated:: 1.0
    Compute the trajectory of a population at some timepoints

    Parameters
    ----------
    timepoints: list
        Containes the ages at which the trajectory is computed

    individual_parameters: IndividualParameters
        Population for which the trajectory should be computed

    leaspy: Leaspy object
        Contains the model parameters

    Returns
    -------
    trajectory: tensor.Tensor
        Contains the trajectory of the population with shape (number of timepoints, number of features)

    Examples
    --------
    >>> leaspy = Leaspy.load(os.path.join(test_data_dir, 'model_parameters', 'test_api.json'))
    >>> ip = IndividualParameters.load(os.path.join(test_data_dir, 'io', 'outputs', 'ip_save.json'))
    >>> timepoints = [70, 71, 72, 73, 74, 75, 76]
    >>> trajectory = compute_trajectory_of_population(timepoints, ip, leaspy)
    """

    warnings.warn('compute_trajectory_of_population function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    if 'univariate' not in leaspy.model.name:
        ip_dict = {k : individual_parameters.get_mean(k) for k in ['xi','tau','sources']}
    else:
        ip_dict = {k : individual_parameters.get_mean(k) for k in ['xi','tau']}

    ip = IndividualParameters()
    ip.add_individual_parameters('mean', ip_dict)
    timepoints = {'mean': timepoints}


    trajectory = leaspy.estimate(timepoints, ip)
    return trajectory['mean']#.reshape(1,trajectory["mean"].shape[0], trajectory["mean"].shape[1])
