
import torch
from leaspy.utils.posterior_analysis.general import compute_trajectory_of_population
from leaspy.utils.posterior_analysis.abnormality import get_age_at_abnormality_conversion
from leaspy.utils.posterior_analysis.statistical_analysis import compute_subgroup_statistics


def compute_trajectory_of_population_resampling(leaspy_iter,
                                                individual_parameters,
                                                timepoints):

    assert len(leaspy_iter)==len(individual_parameters)
    n_resampling_iter = len(leaspy_iter)

    resampling_trajectory = {"average":
    torch.cat([compute_trajectory_of_population(leaspy_iter[resampling_iter],
                                      individual_parameters[resampling_iter],
                                      timepoints)["average"] for resampling_iter in range(n_resampling_iter)],dim=0)}

    return resampling_trajectory


def get_age_at_abnormality_conversion_resampling(leaspy_iter,
                            individual_parameters,
                            timepoints,
                           cutoffs):

    assert len(leaspy_iter)==len(individual_parameters)
    n_resampling_iter = len(leaspy_iter)

    res = torch.cat([get_age_at_abnormality_conversion(leaspy_iter[resampling_iter],
                                                individual_parameters[resampling_iter],
                                                timepoints,
                                                 cutoffs) for resampling_iter in range(n_resampling_iter)],
              dim=0)

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

        difference_subgroups = dict.fromkeys(["mu1", "std1", "mu2", "std2"])
        difference_subgroups["mu"] = mu
        difference_subgroups["std"] = std
        difference_subgroups_resampling[j] = difference_subgroups

    return difference_subgroups_resampling

