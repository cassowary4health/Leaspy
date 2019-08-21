from leaspy.inputs.data.data_reader import DataReader
from leaspy.inputs.data.individual_data import IndividualData
import os
#TODO : object data as output ??? or a result object ? Because there could be ambiguetes here
# TODO or find a good way to say thet there are individual parameters here ???

class Data:
    def __init__(self):

        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.dimension = None
        self.n_individuals = 0
        self.n_visits = 0
        self.n_observations = 0
        self.iter = 0

    def __getitem__(self, idx):
        return self.individuals[self.iter_to_idx[idx]]

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter >= self.n_individuals:
            self.iter = 0
            raise StopIteration
        else:
            self.iter += 1
            return self.__getitem__(self.iter-1)




    @staticmethod
    def from_csv_file(path):
        reader = DataReader(path)

        data = Data()

        data.individuals = reader.individuals
        data.iter_to_idx = reader.iter_to_idx
        data.headers = reader.headers
        data.dimension = reader.dimension
        data.n_individuals = reader.n_individuals
        data.n_visits = reader.n_visits
        data.n_observations = reader.n_observations

        return data



    @staticmethod
    def from_individuals(indices, timepoints, values, headers):
        """
        :param indices: list of indices
        :param timepoints: list of timepoints (nd array)
        :param values: list of values (nd array
        :return:
        """

        data = Data()
        data.dimension = values[0].shape[1]
        data.headers = headers

        for i, idx in enumerate(indices):

            # Create individual
            data.individuals[idx] = IndividualData(idx)
            data.iter_to_idx[data.n_individuals] = idx
            data.n_individuals += 1

            # Add observations / timepoints
            for patient_timepoint_obs, patient_value_obs in zip(timepoints[i], values[i]):
                data.individuals[idx].add_observation(patient_timepoint_obs, patient_value_obs.T.tolist())

            # Update Data metrics
            data.n_visits += len(timepoints)
            data.n_observations += len(timepoints)*data.dimension

        return data
