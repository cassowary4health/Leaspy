from src.inputs.individual_data import IndividualData
import numpy as np

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
        self.time_min = np.nanmin(np.concatenate([individual.timepoints, [self.time_min]]))
        self.time_max = np.nanmax(np.concatenate([individual.timepoints, [self.time_max]]))

    def split(self, indices_train, indices_test):
        data_train = Data()
        data_test = Data()

        for idx_train in indices_train:
            data_train.add_individual(self[idx_train])

        for idx_test in indices_test:
            data_test.add_individual(self[idx_test])

        return data_train, data_test


    def __getitem__(self, id):
         return self.individuals[id]

