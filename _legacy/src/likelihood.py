import numpy as np
import torch

class Likelihood:
    def __init__(self):
        self.individual_attachment = None

    def initialize_likelihood(self, data, model, realizations):
        self.individual_attachment = dict.fromkeys(data.indices)
        #reals_pop, reals_ind = realizations

        # We need the noise to initialize cache variables
        #noise_var = model.compute_sumsquared(data, reals_pop, reals_ind)/(data.n_observations)

        noise_var = (model.compute_sum_squared_tensorized(data, realizations) / (data.n_observations)).sum()

        model.model_parameters['noise_var'] = float(noise_var.detach().numpy())

        # Initialize Cache variables (some depends on noise)
        model._initialize_cache_variables()

        self.update_likelihood(data, model, realizations)


    def update_likelihood(self, data, model, realizations):

        individual_likelihoods = model.compute_individual_attachment_tensorized(data,
                                      realizations)


        for i, idx in enumerate(data.indices):
            self.individual_attachment[idx] = individual_likelihoods[i]


    def __getitem__(self, idx):
        return self.individual_attachment[idx]

    def get_current_attachment(self, indices):
        return np.sum([self.individual_attachment[idx] for idx in indices])

    def set_individual_attachment(self, idx, attachment):
        self.individual_attachment[idx] = attachment
