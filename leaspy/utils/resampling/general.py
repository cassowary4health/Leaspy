
import numpy as np
from leaspy.utils.posterior_analysis.general import compute_trajectory_of_population
from leaspy.utils.posterior_analysis.abnormality import get_age_at_abnormality_conversion
from leaspy.utils.posterior_analysis.statistical_analysis import compute_subgroup_statistics


def compute_trajectory_of_population_resampling(timepoints,
                                                individual_parameters,
                                                leaspy_iter):

    assert len(leaspy_iter)==len(individual_parameters)
    n_resampling_iter = len(leaspy_iter)

    resampling_trajectory = {"mean":
    np.concatenate([compute_trajectory_of_population(timepoints,
                                      individual_parameters[resampling_iter],
                                    leaspy_iter[resampling_iter]) for resampling_iter in range(n_resampling_iter)],axis=0)}

    return resampling_trajectory


def get_age_at_abnormality_conversion_resampling(leaspy_iter,
                            individual_parameters,
                            timepoints,
                           cutoffs):

    assert len(leaspy_iter)==len(individual_parameters)
    n_resampling_iter = len(leaspy_iter)

    res = np.concatenate([get_age_at_abnormality_conversion(cutoffs,
                                                individual_parameters[resampling_iter],
                                                timepoints,
                                                leaspy_iter[resampling_iter]
                                                 ) for resampling_iter in range(n_resampling_iter)],
              axis=0)

    return res




def compute_subgroup_statistics_resampling(leaspy_iter,
                                 individual_parameters_iter,
                                 df_cofactors,
                                 idx_group):

    difference_subgroups_resampling = {}


    for j, (leaspy, individual_parameters) in enumerate(zip(leaspy_iter, individual_parameters_iter)):
        mu, std = compute_subgroup_statistics(leaspy,
                                 individual_parameters,
                                 df_cofactors,
                                 idx_group)

        difference_subgroups = {}
        difference_subgroups["mu"] = mu
        difference_subgroups["std"] = std
        difference_subgroups_resampling[j] = difference_subgroups

    return difference_subgroups_resampling

