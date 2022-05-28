from bisect import bisect

import numpy as np

from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import Any, Dict, FeatureType, IDType, Iterable, List


class IndividualData:
    """
    Container for an individual's data

    Parameters
    ----------
    idx : IDType
        Unique ID
        
    Attributes
    ----------
    idx : IDType
        Unique ID
    timepoints : List[float]
        Timepoints associated with the observations
    observations : np.ndarray[float, 2D]
        Observed data points.
        Shape is ``(n_timepoints, n_features)``
    cofactors : Dict[FeatureType, Any]
        Cofactors in the form {cofactor_name: cofactor_value}
    """

    def __init__(self, idx: IDType):
        self.idx: IDType = idx
        self.timepoints: List[float] = None
        self.observations: np.ndarray = None
        self.cofactors: Dict[FeatureType, Any] = {}

    def add_observations(self, timepoints: List[float], observations: List[List[float]]) -> None:
        """
        Include new observations and associated timepoints

        Parameters
        ----------
        timepoints : List[float]
            Timepoints associated with the observations to include
        observations : array-like[float, 2D]
            Observations to include

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        """
        for i, t in enumerate(timepoints):
            if self.timepoints is None:
                self.timepoints = [timepoints[0]]
                self.observations = np.array([observations[0]])
            elif t in self.timepoints:
                raise LeaspyDataInputError(f'Trying to overwrite timepoint {t}'
                                           f' of individual {self.idx}')
            else:
                index = bisect(self.timepoints, t)
                self.timepoints.insert(index, t)
                self.observations = np.insert(self.observations, index,
                                              observations[i], axis=0)            

    def add_cofactors(self, d: Dict[FeatureType, Any]) -> None:
        """
        Include new cofactors

        Parameters
        ----------
        d : Dict[FeatureType, Any]
            Cofactors to include, in the form `{name: value}`

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        """
        if not (
            isinstance(d, dict)
            and all(isinstance(k, str) for k in d.keys())
        ):
            raise TypeError("Invalid argument type for `d`")

        for k, v in d.items():
            if k in self.cofactors.keys() and v != self.cofactors[k]:
                raise LeaspyDataInputError(f"Cofactor {k} is already present"
                                           f" for patient {self.idx}")
            self.cofactors[k] = v
