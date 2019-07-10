import numpy as np


class Dataset:
    def __init__(self, data, model=None, algo=None):

        self.values = None
        self.mask = None
        self.n_individuals = None
        self.max_observations = None
        self.dimension = None

        if model is not None:
            self._check_model_compatibility(data, model)
        if algo is not None:
            self._check_algo_compatibility(data, algo)

        self._construct_values(data)

    def _construct_values(self, data):

        batch_size = data.n_individuals
        x_len = [len(_.timepoints) for _ in data]
        channels = data.dimension
        values = np.zeros((batch_size, max(x_len), channels))
        mask = np.zeros((batch_size, max(x_len), 1))

        for i, d in enumerate(x_len):
            indiv_values = data[i].observations
            values[i, 0:d, :] = indiv_values
            mask[i, 0:d, :] = 1

        self.n_individuals = batch_size
        self.max_observations = max(x_len)
        self.dimension = channels
        self.values = values
        self.mask = mask

    def _check_model_compatibility(self, data, model):
        if model.dimension is None:
            return
        if data.dimension != model.dimension:
            raise ValueError("The initialized model and the data do not have the same dimension")

    def _check_algo_compatibility(self, data, algo):
        #TODO
        return 0
