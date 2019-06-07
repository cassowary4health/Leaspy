from src.inputs.individual_data import IndividualData
import numpy as np
import pandas as pd

class Data():
    def __init__(self):
        self.indices = []
        self.individuals = {}

        # Metrics
        self.n_individuals = 0
        self.n_visits = 0
        self.n_observations = 0
        self.time_min = np.nan
        self.time_max = np.nan
        self.dimension = None

    def add_individual(self, individual):
        if individual.idx in self.indices:
            raise ValueError("There already")

        self.indices.append(individual.idx)
        self.individuals[individual.idx] = individual

        self.update_metrics(individual)

    def update_metrics(self, individual):
        self.n_individuals += 1
        self.n_visits += individual.n_visits
        self.n_observations += individual.n_observations

    def split(self, indices_train, indices_test):
        data_train = Data()
        data_test = Data()

        for idx_train in indices_train:
            data_train.add_individual(self[idx_train])

        for idx_test in indices_test:
            data_test.add_individual(self[idx_test])

        return data_train, data_test


    def set_time_normalization_info(self, time_mean, time_std, time_min, time_max):
        self.time_mean = time_mean
        self.time_std = time_std
        self.time_min = time_min
        self.time_max = time_max

    def __getitem__(self, id):
         return self.individuals[id]

    def set_dimension(self, dimension):
        self.dimension = dimension

    def to_pandas(self):

        df = pd.DataFrame()
        for idx in self.indices:
            times = self[idx].tensor_timepoints
            x = self[idx].tensor_observations
            df_patient = pd.DataFrame(data=x.detach().numpy(), index=times.detach().numpy().reshape(-1))
            df_patient = df_patient.add_prefix('value_')
            df_patient.index.name = 'TIMES'
            df_patient['ID'] = idx
            df = pd.concat([df, df_patient])

        return df.reset_index().set_index(['ID', 'TIMES'])


