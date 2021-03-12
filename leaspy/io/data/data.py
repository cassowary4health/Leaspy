import numpy as np
import pandas as pd
import torch

from leaspy.io.data.csv_data_reader import CSVDataReader
from leaspy.io.data.dataframe_data_reader import DataframeDataReader
from leaspy.io.data.individual_data import IndividualData


# from leaspy.io.data.dataset import Dataset


# TODO : object data as logs ??? or a result object ? Because there could be ambiguetes here
# TODO or find a good way to say thet there are individual parameters here ???

class Data:
    """
    Main data container, initialized from a `csv file` or a :class:`pandas.DataFrame`.
    """
    def __init__(self):

        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.dimension = None
        self.n_individuals = 0
        self.n_visits = 0
        self.cofactors = []

        self.iter = 0

    def get_by_idx(self, idx):
        """
        Get the :class:`~leaspy.io.data.individual_data.IndividualData` of a an individual identified by its ID.

        Returns
        -------
        :class:`~leaspy.io.data.individual_data.IndividualData`
        """
        return self.individuals[idx]

    def __getitem__(self, iter):
        return self.individuals[self.iter_to_idx[iter]]

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter >= self.n_individuals:
            self.iter = 0
            raise StopIteration
        else:
            self.iter += 1
            return self.__getitem__(self.iter - 1)

    def load_cofactors(self, df, cofactors):
        """
        Load cofactors from a `pandas.DataFrame` to the `Data` object

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            the index is the list of subject ids
        cofactors: list[str]
            names of the column(s) of df which shall be loaded as cofactors
        """

        df = df.copy(deep=True)

        for iter, idx in self.iter_to_idx.items():
            # Get the cofactors and check that it is unique
            try:
                cof = df.loc[[idx]][cofactors].to_dict(orient='list')
            except KeyError:
                # If the ID are for example '116' - pandas save & reload it as integer & might induce errors
                cof = df.loc[[int(idx)]][cofactors].to_dict(orient='list')

            for c in cofactors:
                v = np.unique(cof[c])
                v = [_ for _ in v if _ == _]
                if len(v) > 1:
                    raise ValueError("Multiples values of the cofactor {} for patient {} : {}".format(c, idx, v))
                elif len(v) == 0:
                    cof[c] = None
                else:
                    cof[c] = v[0]

            # Add these cofactor to the individual
            self.individuals[idx].add_cofactors(cof)
        self.cofactors += cofactors

    @staticmethod
    def from_csv_file(path):
        """
        Create a `Data` object from a CSV file.

        Returns
        -------
        `Data`
        """
        reader = CSVDataReader(path)
        return Data._from_reader(reader)

    def to_dataframe(self, cofactors=None):
        """
        Return the subjects' observations in a :class:`pandas.DataFrame` along their ID and ages at all visits.

        Parameters
        ----------
        cofactors : str, list [str], optional (default None)
            Contains the cofactors' names to be included in the DataFrame. By default, no cofactors are returned.
            If cofactors == "all", all the available cofactors are returned.

        Returns
        -------
        :class:`pandas.DataFrame`
            Contains the subjects's ID, age and scores (optional - and cofactors) for each timepoint.
        """
        indices = []
        timepoints = torch.zeros((self.n_visits, 1))
        arr = torch.zeros((self.n_visits, self.dimension))

        iteration = 0
        for i, indiv in enumerate(self.individuals.values()):
            ages = indiv.timepoints
            for j, age in enumerate(ages):
                indices.append(indiv.idx)
                timepoints[iteration] = age
                arr[iteration] = torch.tensor(indiv.observations[j], dtype=torch.float32)

                iteration += 1

        arr = torch.cat((timepoints, arr), dim=1).tolist()

        df = pd.DataFrame(data=arr, index=indices, columns=['TIME'] + self.headers)
        df.index.name = 'ID'

        if cofactors is not None:
            if type(cofactors) == str:
                if cofactors == "all":
                    cofactors_list = self.cofactors
            else:
                cofactors_list = cofactors
            for cofactor in cofactors_list:
                df[cofactor] = ''
                for subject_name in indices:
                    df.loc[subject_name, cofactor] = self.individuals[subject_name].cofactors[cofactor]

        return df.reset_index()

    @staticmethod
    def from_dataframe(df):
        """
        Create a `Data` object from a :class:`pandas.DataFrame`.

        Returns
        -------
        `Data`
        """
        reader = DataframeDataReader(df)
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
    def from_individuals(indices, timepoints, values, headers):
        """
        Create a `Data` class object from lists of `ID`, `timepoints` and the corresponding `values`.

        Parameters
        ----------
        indices: list[str]
            Contains the individuals' ID.
        timepoints: list[array-like 1D]
            For each individual ``i``, list of ages at visits.
            Number of timepoints is refered below as ``n_timepoints_i``
        values: list[array-like 2D]
            For each individual ``i``, all values at visits.
            Shape is ``(n_timepoints_i, n_features)``.
        headers: list[str]
            Contains the features' names.

        Returns
        -------
        `Data`
            Data class object with all ID, timepoints, values and feautures' names.
        """
        data = Data()
        data.dimension = len(values[0][0])
        data.headers = headers

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
