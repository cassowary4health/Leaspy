import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

def append_spaceshifts_to_individual_parameters_dataframe(df_individual_parameters, leaspy):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('append_spaceshifts_to_individual_parameters_dataframe function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    # TODO: Igor test
    df_ip = df_individual_parameters.copy()

    sources = df_ip [['sources_' + str(i) for i in range(leaspy.model.source_dimension)]].values.T
    spaceshifts = np.dot(leaspy.model.attributes.mixing_matrix, sources)

    for i, spaceshift_coord in enumerate(spaceshifts):
        #population_speed_coord = np.exp(float((leaspy.model.parameters["v0"][i])))
        df_ip['w_' + str(i)] = spaceshift_coord#/population_speed_coord

    return df_ip


def compute_subgroup_statistics(leaspy,
                                 individual_parameters,
                                 df_cofactors,
                                 idx_group):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('compute_subgroup_statistics function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    df_indparam = append_spaceshifts_to_individual_parameters_dataframe(individual_parameters.to_dataframe(),
                                                                        leaspy)
    df_run = pd.concat([df_indparam, df_cofactors], axis=1, sort=True)

    mu_grp = df_run.loc[idx_group].mean()
    std_grp = df_run.loc[idx_group].std()
    return mu_grp, std_grp


def compute_correlation(leaspy, individual_parameters, df_cofactors, method="pearson"):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('compute_correlation function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    df_indparam = append_spaceshifts_to_individual_parameters_dataframe(individual_parameters.to_dataframe(), leaspy)

    # Compute PCA of w_i
    if "sources_0" in df_indparam.columns:
        pca = PCA(n_components=2)
        w_features = [feature for feature in df_indparam if "w_" in feature]
        pca.fit(df_indparam[w_features])

        print("PCA variance ratio : ", pca.explained_variance_ratio_)

        res_pca = pca.transform(df_indparam[w_features])

        df_indparam["w_pca1"] = res_pca[:, 0]
        df_indparam["w_pca2"] = res_pca[:, 1]

    df = pd.concat([df_indparam, df_cofactors], axis=1, sort=True).dropna()

    if method =="pearson":
        correlation_function = stats.pearsonr
    elif method == "spearman":
        correlation_function = stats.spearmanr
    else:
        raise ValueError("Correlation not known")

    # P-values
    features = df.columns

    features = [feature for feature in df_indparam.columns if "sources" not in feature]
    if "sources_0" in df.columns:
        features += ["sources"]
    features += list(df_cofactors.columns)

    # Create
    df_corr_value = pd.DataFrame(np.nan*np.zeros(shape=(len(features),len(features))), index=features, columns=features)
    df_corr_logpvalue = pd.DataFrame(np.nan*np.zeros(shape=(len(features),len(features))), index=features, columns=features)

    p = len(features)

    for i in range(p):
        for j in range(i):
            feature_row = features[i]
            feature_col = features[j]

            # Compute Correlations
            # Remove w_i between them
            #if "w_" in feature_row and "w_" in feature_col:
            #    value, pvalue = np.nan, np.nan
            if feature_row != 'sources' and feature_col != 'sources':
                df_corr = df[[feature_row, feature_col]].dropna()
                value, pvalue = correlation_function(df_corr.iloc[:, 0], df_corr.iloc[:, 1])
            else:
                features_source = [feature for feature in df.columns if "sources" in feature]
                feature_not_source = feature_row if "sources" in feature_col else feature_col
                if "w_" not in feature_not_source:
                    sampling_size = df[features_source].shape[0]
                    X_train = df[features_source].values[:int(sampling_size/2)]
                    X_test = df[features_source].values[int(sampling_size/2):]
                    Y_train = df[feature_not_source].values[:int(sampling_size/2)]
                    Y_test = df[feature_not_source].values[int(sampling_size/2):]
                    pls2 = PLSRegression(n_components=1)
                    pls2.fit(X_train, Y_train)
                    Y_pred = pls2.predict(X_test)
                    df_corr = pd.DataFrame(np.array([Y_pred.reshape(-1),  Y_test.reshape(-1)]).T)
                    value, pvalue = correlation_function(df_corr.iloc[:,0], df_corr.iloc[:,1])
                else:
                    value , pvalue = np.nan, np.nan

            logpvalue = np.log10(pvalue)

            df_corr_logpvalue.iloc[i, j] = logpvalue
            df_corr_value.iloc[i, j] = value
            df_corr_logpvalue.iloc[j, i] = logpvalue
            df_corr_value.iloc[j, i] = value



    return df_corr_value, df_corr_logpvalue
