import warnings
import numpy as np


from leaspy.utils.posterior_analysis.general import compute_trajectory_of_population


def get_age_at_abnormality_conversion(abnormality_thresholds,
                                      individual_parameter,
                                      timepoints,
                                      leaspy):
    """
    .. deprecated:: 1.0
    """
    warnings.warn('get_age_at_abnormality_conversion function is deprecated. Please use the one in Leaspype', DeprecationWarning)

    features = leaspy.model.features

    # compute model values
    model_values = compute_trajectory_of_population(timepoints,
                                                    individual_parameter,
                                                    leaspy
                                                    )

    # Get cutoffs
    cutoffs = [abnormality_thresholds[feature] for feature in features]

    # At which index do we reach threshold ?
    is_superior_cutoff = model_values >= np.array(cutoffs)

    # Translate to times
    times_reach_list = []
    for j, feature in enumerate(features):
        where_reach = np.where(np.diff(is_superior_cutoff[:, j], axis=0))[0]
        len_idx = len(where_reach)
        # If threshold is reached
        if len_idx == 1:
            idx_reached = int(where_reach)
            if idx_reached > 0:
                # Interpolate between before and now
                times_reach = 0.5 * (timepoints[idx_reached] + timepoints[idx_reached - 1])
            else:
                # Keep only first
                times_reach = timepoints[idx_reached]
        # If threhold was not reached
        elif len_idx == 0:
            print("Warning : cutoff for feature {} at index {} was not reached".format(feature, j))
            # If always below
            if is_superior_cutoff[0, 0, j]:
                times_reach = timepoints[0]
            # If always above
            else:
                times_reach = timepoints[-1]
        else:
            raise ValueError("Threshold reached at multiples times")
        times_reach_list.append(np.array(times_reach).reshape(1,1))
    res = np.concatenate(times_reach_list, axis=1)
    #res = np.expand_dims(res, axis=0)

    return res
