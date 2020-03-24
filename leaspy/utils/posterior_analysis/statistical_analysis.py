import numpy as np



def append_spaceshifts_to_individual_parameters_dataframe(df_individual_parameters, leaspy):
    # TODO: Igor test
    df_ip = df_individual_parameters.copy()

    sources = df_ip [['sources_' + str(i) for i in range(leaspy.model.source_dimension)]].values.T
    spaceshifts = np.dot(leaspy.model.attributes.mixing_matrix, sources)

    for i, spaceshift_coord in enumerate(spaceshifts):
        df_ip['w_' + str(i)] = spaceshift_coord

    return df_ip

