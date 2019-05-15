import numpy as np

class Likelihood:
    def __init__(self):
        self.individual_attachment = None

    def _initialize_likelihood(self, data, model, realizations):
        self.individual_attachment = dict.fromkeys(data.indices)
        reals_pop, reals_ind = realizations

        for idx in data.indices:
            self.individual_attachment[idx] = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])

    def __getitem__(self, idx):
        return self.individual_attachment[idx]

    def compute_current_attachment(self):
        return np.sum(list(self.individual_attachment.values()))
