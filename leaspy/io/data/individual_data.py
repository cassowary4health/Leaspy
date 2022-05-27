from bisect import bisect

import numpy as np

from leaspy.exceptions import LeaspyDataInputError, LeaspyInputError
from leaspy.utils.typing import IDType, DictParams, KwargsType, List, Iterable


class IndividualData:
    """
    Data container for individual parameters, used to construct a `Data` container.

    Parameters
    ----------
    idx : str
        The identifier of subject.

    Raises
    ------
    :exc:`.LeaspyDataInputError`
    """

    def __init__(self, idx: IDType):
        self.idx = idx
        self.timepoints: List[float] = None
        self.observations: Iterable[Iterable[float]] = None
        self.individual_parameters: DictParams = {}
        self.cofactors: KwargsType = {}

    def add_observations(self, timepoints: List[float],
                         observations: List[List[float]]) -> None:
        """
        Include new observations and associated timepoints

        Parameters
        ----------
        timepoints : List[float]
            Timepoints associated with the observations
        observations : List[List[float]]
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

    def add_individual_parameters(self, name, value):
        self.individual_parameters[name] = value

    def add_cofactors(self, d: dict):
        for k, v in d.items():
            if k in self.cofactors.keys() and v != self.cofactors[k]:
                raise LeaspyDataInputError(f"The cofactor {k} is already present for patient {self.idx}")
            self.cofactors[k] = v

    def add_cofactor(self, name, value):
        self.cofactors[name] = value
