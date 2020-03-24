import numpy as np




def get_reparametrized_ages(individual_parameters, diagnosis_ages, leaspy):
    r"""
    Description

    Parameters
    ----------
    individual_parameters: Individual parameters object
        Contains the individual parameters for each patient

    diagnosis_ages: dict {patient_idx: [age_at_diagnosis]}
        Contains the ages at which each patient has converted

    leaspy: Leaspy object
        Contains the model parameters

    Returns
    -------

    Examples
    --------

    >>> leaspy =
    >>> ...
    """

    # TODO IGor

    reparametrized_ages = {}

    for idx, age_at_diag in diagnosis_ages.values():
        # TODO
        idx_ip = individual_parameters[idx]
        alpha = np.exp(idx_ip['xi'])
        tau = idx_ip['tau']

        reparam_diag = alpha * (age_at_diag - tau ) + leaspy.model.parameters['tau_mean']


    return reparametrized_ages


def compute_trajectory_of_population(leaspy, individual_parameters, timepoints):


    dict = {
        'xi': individual_parameters.get_mean('xi'),
        'tau': individual_parameters.get_mean('tau'),
        'sources': individual_parameters
    }

    ip = IndividualParameters()
    ip.add_ip('mean', dict)

    leaspy.estimate()