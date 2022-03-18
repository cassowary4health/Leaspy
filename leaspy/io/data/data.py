import warnings

import numpy as np
import pandas as pd
import torch

from leaspy.io.data.csv_data_reader import CSVDataReader
from leaspy.io.data.dataframe_data_reader import DataframeDataReader
from leaspy.io.data.individual_data import IndividualData
# from leaspy.io.data.dataset import Dataset

from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import FeatureType, IDType, Dict, List

# TODO : object data as logs ??? or a result object ? Because there could be ambiguities here
# TODO or find a good way to say that there are individual parameters here ???


class Data:
    """
    Main data container, initialized from a `csv file` or a :class:`pandas.DataFrame`.
    """
    def __init__(self):

        self.individuals: Dict[IDType, IndividualData] = {}
        self.iter_to_idx: Dict[int, IDType] = {}
        self.headers: List[FeatureType] = None
        self.dimension: int = None
        self.n_individuals: int = 0
        self.n_visits: int = 0
        self.cofactors: List[FeatureType] = []

        self.iter: int = 0

    def get_by_idx(self, idx: IDType):
        """
        Get the :class:`~leaspy.io.data.individual_data.IndividualData` of a an individual identified by its ID.

        Parameters
        ----------
        idx : IDType
            The identifier of the patient you want to get the individual data.

        Returns
        -------
        :class:`~leaspy.io.data.individual_data.IndividualData`
        """
        return self.individuals[idx]

    def __getitem__(self, iter: int):
        return self.individuals[self.iter_to_idx[iter]]

    def __iter__(self):
        # TODO: make a true DataIterator class because quite dirty to have `iter` inside
        return self

    def __next__(self):
        # TODO: make a true DataIterator class because quite dirty to have `iter` inside
        if self.iter >= self.n_individuals:
            self.iter = 0
            raise StopIteration
        else:
            self.iter += 1
            return self.__getitem__(self.iter - 1)

    def load_cofactors(self, df: pd.DataFrame, *, cofactors: List[FeatureType] = None):
        """
        Load cofactors from a `pandas.DataFrame` to the `Data` object

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            The dataframe where the cofactors are stored.
            Its index should be ID, the identifier of subjects
            and it should uniquely index the dataframe (i.e. one row per individual).
        cofactors : list[str] or None (default)
            Names of the column(s) of df which shall be loaded as cofactors.
            If None, all the columns from the input dataframe will be loaded as cofactors.

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        """

        if not (
            isinstance(df, pd.DataFrame)
            and isinstance(df.index, pd.Index)
            and df.index.names == ["ID"]
            and df.index.notnull().all()
            and df.index.is_unique
        ):
            raise LeaspyDataInputError("You should pass a dataframe whose index ('ID') should "
                                       "not contain any NaN nor any duplicate.")

        internal_dtype_indices = pd.api.types.infer_dtype(self.iter_to_idx.values())
        cofactors_dtype_indices = pd.api.types.infer_dtype(df.index)
        if cofactors_dtype_indices != internal_dtype_indices:
            raise LeaspyDataInputError(f"The ID type in your cofactors ({cofactors_dtype_indices}) "
                                       f"is inconsistent with the ID type in Data ({internal_dtype_indices}):\n{df.index}")

        internal_indices = pd.Index(self.iter_to_idx.values())
        individuals_without_cofactors = internal_indices.difference(df.index)
        unknown_individuals = df.index.difference(internal_indices)

        if len(individuals_without_cofactors):
            raise ValueError(f"These individuals do not have cofactors: {individuals_without_cofactors}")
        if len(unknown_individuals):
            warnings.warn(f"These individuals with cofactors are not part of your Data: {unknown_individuals}")

        if cofactors is None:
            cofactors = df.columns.tolist()

        # sub-select the individuals & cofactors to look for
        d_cofactors = df.loc[internal_indices, cofactors].to_dict(orient='index')

        # Loop per individual
        for idx_subj, d_cofactors_subj in d_cofactors.items():
            self.individuals[idx_subj].add_cofactors(d_cofactors_subj)

        self.cofactors += cofactors

    @staticmethod
    def from_csv_file(path: str, **kws):
        """
        Create a `Data` object from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file to load (with extension)
        **kws
            Keyword arguments that are sent to :class:`.CSVDataReader`

        Returns
        -------
        :class:`.Data`
        """
        reader = CSVDataReader(path, **kws)
        return Data._from_reader(reader)

    def to_dataframe(self, *, cofactors=None):
        """
        Return the subjects' observations in a :class:`pandas.DataFrame` along their ID and ages at all visits.

        Parameters
        ----------
        cofactors : list[str], 'all', or None (default None)
            Contains the cofactors' names to be included in the DataFrame.
            If None (default), no cofactors are returned.
            If "all", all the available cofactors are returned.

        Returns
        -------
        :class:`pandas.DataFrame`
            Contains the subjects' ID, age and scores (optional - and cofactors) for each timepoint.

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        """
        indices = []
        timepoints = torch.zeros((self.n_visits, 1))
        arr = torch.zeros((self.n_visits, self.dimension))

        iteration = 0
        for indiv in self.individuals.values():
            ages = indiv.timepoints
            for j, age in enumerate(ages):
                indices.append(indiv.idx)
                timepoints[iteration] = age
                # TODO: IndividualData.observations is really badly constructed (list of numpy 1D arrays), we should change this...
                arr[iteration] = torch.tensor(np.array(indiv.observations[j]), dtype=torch.float32)

                iteration += 1

        arr = torch.cat((timepoints, arr), dim=1).tolist()

        df = pd.DataFrame(data=arr, index=indices, columns=['TIME'] + self.headers)
        df.index.name = 'ID'

        if cofactors is not None:
            cofactors_list = None

            if isinstance(cofactors, str) and cofactors == "all":
                cofactors_list = self.cofactors
            elif isinstance(cofactors, list):
                cofactors_list = cofactors

            if cofactors_list is None:
                raise LeaspyDataInputError("`cofactor` should either be 'all' or a list[str]")

            for cofactor in cofactors_list:
                df[cofactor] = ''
                for subject_name in indices:
                    df.loc[subject_name, cofactor] = self.individuals[subject_name].cofactors[cofactor]

        return df.reset_index()

    @staticmethod
    def from_dataframe(df: pd.DataFrame, **kws):
        """
        Create a `Data` object from a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Dataframe containing ID, TIME and features.
        **kws
            Keyword arguments that are sent to :class:`.DataframeDataReader`

        Returns
        -------
        `Data`
        """
        reader = DataframeDataReader(df, **kws)
        return Data._from_reader(reader)

    @staticmethod
    def _from_reader(reader):
        data = Data()

        data.individuals = reader.individuals
        data.iter_to_idx = reader.iter_to_idx
        data.headers = reader.headers
        data.dimension = reader.dimension
        data.n_individuals = reader.n_individuals
        data.n_visits = reader.n_visits

        return data

    @staticmethod
    def from_individuals(indices: List[IDType], timepoints: List[List], values: List[List], headers: List[FeatureType]):
        """
        Create a `Data` class object from lists of `ID`, `timepoints` and the corresponding `values`.

        Parameters
        ----------
        indices : list[str]
            Contains the individuals' ID.
        timepoints : list[array-like 1D]
            For each individual ``i``, list of ages at visits.
            Number of timepoints is referred below as ``n_timepoints_i``
        values : list[array-like 2D]
            For each individual ``i``, all values at visits.
            Shape is ``(n_timepoints_i, n_features)``.
        headers : list[str]
            Contains the features' names.

        Returns
        -------
        `Data`
            Data class object with all ID, timepoints, values and features' names.
        """
        data = Data()

        data.headers = headers
        data.dimension = len(headers)

        for i, idx in enumerate(indices):
            # Create individual
            data.individuals[idx] = IndividualData(idx)
            data.iter_to_idx[data.n_individuals] = idx
            data.n_individuals += 1

            # Add observations / timepoints
            data.individuals[idx].add_observations(timepoints[i], values[i])

            # Update Data metrics
            data.n_visits += len(timepoints[i])

        return data
