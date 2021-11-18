from bisect import bisect

from leaspy.exceptions import LeaspyDataInputError
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
        self.observations: List[Iterable[float]] = None
        self.individual_parameters: DictParams = {}
        self.cofactors: KwargsType = {}

    def add_observation(self, timepoint, observation):
        if self.timepoints is None:
            self.timepoints = []
            self.observations = []

        if timepoint in self.timepoints:
            raise LeaspyDataInputError(
                f'You are trying to overwrite the observation at time {timepoint} of the subject {self.idx}')

        index = bisect(self.timepoints, timepoint)
        self.timepoints.insert(index, timepoint)
        self.observations.insert(index, observation)

    def add_observations(self, timepoints, observations):

        for i, timepoint in enumerate(timepoints):
            self.add_observation(timepoint, observations[i])

    def add_individual_parameters(self, name, value):
        self.individual_parameters[name] = value

    def add_cofactors(self, d: dict):
        for k, v in d.items():
            if k in self.cofactors.keys() and v != self.cofactors[k]:
                raise LeaspyDataInputError(f"The cofactor {k} is already present for patient {self.idx}")
            self.cofactors[k] = v

    def add_cofactor(self, name, value):
        self.cofactors[name] = value
