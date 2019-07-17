from src.inputs.data.data_reader import DataReader


class Data:
    def __init__(self, path):
        reader = DataReader(path)

        self.individuals = reader.individuals
        self.iter_to_idx = reader.iter_to_idx
        self.headers = reader.headers
        self.dimension = reader.dimension
        self.n_individuals = reader.n_individuals
        self.n_visits = reader.n_visits
        self.n_observations = reader.n_observations

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
