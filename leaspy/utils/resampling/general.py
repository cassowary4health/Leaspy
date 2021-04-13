import warnings
import numpy as np
from leaspy.utils.posterior_analysis.general import compute_trajectory_of_population
from leaspy.utils.posterior_analysis.abnormality import get_age_at_abnormality_conversion
from leaspy.utils.posterior_analysis.statistical_analysis import compute_subgroup_statistics
from leaspy.utils.posterior_analysis.statistical_analysis import compute_correlation

import pandas as pd

def compute_trajectory_of_population_resampling(timepoints,
                                                individual_parameters,
                                                leaspy_iter):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('compute_trajectory_of_population_resampling function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    assert len(leaspy_iter)==len(individual_parameters)
    n_resampling_iter = len(leaspy_iter)

    resampling_trajectory = np.array([compute_trajectory_of_population(timepoints,
                                      individual_parameters[resampling_iter],
                                    leaspy_iter[resampling_iter]) for resampling_iter in range(n_resampling_iter)])

    return resampling_trajectory


def get_age_at_abnormality_conversion_resampling(leaspy_iter,
                            individual_parameters,
                            timepoints,
                           cutoffs):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('get_age_at_abnormality_conversion_resampling function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    assert len(leaspy_iter)==len(individual_parameters)
    n_resampling_iter = len(leaspy_iter)

    res = np.array([get_age_at_abnormality_conversion(cutoffs,
                                                individual_parameters[resampling_iter],
                                                timepoints,
                                                leaspy_iter[resampling_iter]
                                                 ) for resampling_iter in range(n_resampling_iter)])

    return res




def compute_subgroup_statistics_resampling(leaspy_iter,
                                 individual_parameters_iter,
                                 df_cofactors,
                                 idx_group):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('compute_subgroup_statistics_resampling function is deprecated. Please use the one in Leaspype', DeprecationWarning)

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



def compute_correlation_resampling(leaspy_iter, individual_parameters_iter, df_cofactors_dummy):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('compute_correlation_resampling function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    assert len(leaspy_iter) == len(individual_parameters_iter)

    corr_value_dict = {}
    corr_log10pvalue_dict = {}


    for i, (leaspy, individual_parameters) in enumerate(zip(leaspy_iter, individual_parameters_iter)):
        corr_value, corr_log10pvalue = compute_correlation(leaspy, individual_parameters,
                                                           df_cofactors_dummy)

        corr_value_dict[i] = corr_value
        corr_log10pvalue_dict[i] = corr_log10pvalue


    #features = corr_value.columns
    #df_value_concat = pd.concat(list(corr_value_dict.values()))
    #value_by_row_index = df_value_concat.groupby(df_value_concat.index)
    #corr_value_mean = value_by_row_index.mean().loc[features, features]
    #corr_value_std = value_by_row_index.mean().loc[features, features]

    #df_log10pvalue_concat = pd.concat(list(corr_log10pvalue_dict.values()))
    #log10pvalue_by_row_index = df_log10pvalue_concat.groupby(df_log10pvalue_concat.index)
    #corr_log10pvalue_mean = log10pvalue_by_row_index.mean().loc[features, features]
    #corr_log10pvalue_95percent = log10pvalue_by_row_index.quantile(0.95).loc[features, features]
    #corr_log10pvalue_std = log10pvalue_by_row_index.std().loc[features, features]

    return corr_value_dict, corr_log10pvalue_dict




