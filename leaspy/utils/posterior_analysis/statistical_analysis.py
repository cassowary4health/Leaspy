import numpy as np
import pandas as pd
import scipy.stats as stats

def append_spaceshifts_to_individual_parameters_dataframe(df_individual_parameters, leaspy):
    # TODO: Igor test
    df_ip = df_individual_parameters.copy()

    sources = df_ip [['sources_' + str(i) for i in range(leaspy.model.source_dimension)]].values.T
    spaceshifts = np.dot(leaspy.model.attributes.mixing_matrix, sources)

    for i, spaceshift_coord in enumerate(spaceshifts):
        population_speed_coord = np.exp(float((leaspy.model.parameters["v0"][i])))
        df_ip['w_' + str(i)] = spaceshift_coord/population_speed_coord

    return df_ip


def compute_subgroup_statistics(leaspy,
                                 individual_parameters,
                                 df_cofactors,
                                 idx_group):

    df_indparam = append_spaceshifts_to_individual_parameters_dataframe(individual_parameters.to_dataframe(),
                                                                        leaspy)
    df_run = pd.concat([df_indparam, df_cofactors], axis=1, sort=True)

    mu_grp = df_run.loc[idx_group].mean()
    std_grp = df_run.loc[idx_group].std()
    return mu_grp, std_grp


def compute_correlation(leaspy, individual_parameters, df_cofactors, method="pearson"):

    df_indparam = append_spaceshifts_to_individual_parameters_dataframe(individual_parameters.to_dataframe(), leaspy)
    df = pd.concat([df_indparam, df_cofactors], axis=1, sort=True)

    df_corr_value = df.corr(method=method)*np.nan
    df_corr_logpvalue = df_corr_value.copy(deep=True)*np.nan

    if method =="pearson":
        correlation_function = stats.pearsonr
    elif method == "spearman":
        correlation_function = stats.spearmanr
    else:
        raise ValueError("Correlation not known")

    # P-values
    features = df.columns
    p = len(df.columns)

    for i in range(p):
        for j in range(i):
            feature_row = features[i]
            feature_col = features[j]

            # Compute Correlations
            df_corr = df[[feature_row, feature_col]].dropna()
            value, pvalue = correlation_function(df_corr[feature_row], df_corr[feature_col])
            logpvalue = np.log10(pvalue)

            df_corr_logpvalue.iloc[i, j] = logpvalue
            df_corr_value.iloc[i, j] = value
            df_corr_logpvalue.iloc[j, i] = logpvalue
            df_corr_value.iloc[j, i] = value

    return df_corr_value, df_corr_logpvalue