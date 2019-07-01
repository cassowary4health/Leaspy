import numpy as np

class Likelihood:
    def __init__(self):
        self.individual_attachment = None

    def initialize_likelihood(self, data, model, realizations):
        self.individual_attachment = dict.fromkeys(data.indices)
        reals_pop, reals_ind = realizations

        # We need the noise to initialize cache variables
        noise_var = model.compute_sumsquared(data, reals_pop, reals_ind)/(model.dimension*data.n_observations)
        model.model_parameters['noise_var'] = float(noise_var.detach().numpy())

        # Initialize Cache variables (some depends on noise)
        model._initialize_cache_variables()

        for idx in data.indices:
            self.individual_attachment[idx] = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])



    def __getitem__(self, idx):
        return self.individual_attachment[idx]

    def get_current_attachment(self):
        return np.sum(list(self.individual_attachment.values()))
