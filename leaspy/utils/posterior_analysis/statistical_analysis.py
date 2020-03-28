import numpy as np
import pandas as pd

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


def compute_correlation(leaspy, individual_parameters, df_cofactors):

    df_indparam = append_spaceshifts_to_individual_parameters_dataframe(individual_parameters.to_dataframe(), leaspy)
    df = pd.concat([df_indparam, df_cofactors], axis=1, sort=True)
    return 0