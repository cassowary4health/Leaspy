import numpy as np
import torch
from .abstract_sampler import AbstractSampler
import itertools

class GibbsSampler(AbstractSampler):

    def __init__(self, info,n_patients):
        super().__init__(info,n_patients)
        #AbstractSampler.__init__(self,info,n_patients)

        self.std = None

        if info["type"] == "population":
            self.std = 0.005
        elif info["type"] == "individual":
            self.std = 0.1

        # Acceptation rate
        self.counter_acceptation = 0
        # Torch distribution
        self.distribution = torch.distributions.normal.Normal(loc=0.0, scale = self.std)

    def sample(self, data, model, realizations,temperature_inv):
        if self.type == 'pop':
            self._sample_population_realizations(data, model, realizations,temperature_inv)
        else:
            self._sample_individual_realizations(data, model, realizations,temperature_inv)

    def _proposal(self,val):
        return val+self.distribution.sample(sample_shape=val.shape)


    def _update_std(self,accepted):
        self.counter_acceptation += len(accepted)

        if self.counter_acceptation == self.temp_length:
            # Update the std of sampling so that expected rate is reached
            if np.mean(self.acceptation_temp) < 0.2:
                self._set_std(0.9 * self.std)
                #self.std = 0.9 * self.std
                #print("Decreased std of sampler-{0}".format(self.name))
            elif np.mean(self.acceptation_temp) > 0.4:
                self._set_std(1.1 * self.std)
                #self.std = 1.1 * self.std
                #print("Increased std of sampler-{0}".format(self.name))

            # reset acceptation temp list
            self.counter_acceptation = 0

    def _set_std(self, std):
        self.std = std
        self.distribution = torch.distributions.normal.Normal(loc=0.0, scale=std)

    def _sample_population_realizations(self, data, model, realizations,temperature_inv):
        shape_current_variable = realizations[self.name].shape
        index = [e for e in itertools.product(*[range(s) for s in shape_current_variable])]
        for idx in index:
            # Compute the attachment and regularity
            previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
            previous_regularity = model.compute_regularity_variable(realizations[self.name])

            # Keep previous realizations and sample new ones
            previous_reals_pop = realizations[self.name].tensor_realizations.clone()
            new_val = self._proposal(realizations[self.name].tensor_realizations[idx])
            realizations[self.name].set_tensor_realizations_element(new_val,idx)

            # Update intermediary model variables if necessary
            model.update_MCMC_toolbox([self.name], realizations)

            # Compute the attachment and regularity
            new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
            new_regularity = model.compute_regularity_variable(realizations[self.name])
            alpha = torch.exp(-((new_regularity.sum() - previous_regularity.sum()) * temperature_inv +
                                (new_attachment - previous_attachment)))

            accepted = self._metropolis_step(alpha)
            self._update_acceptation_rate([accepted])
            self._update_std([accepted])

            # Revert if not accepted
            if not accepted:
                # Revert realizations
                realizations[self.name].tensor_realizations = previous_reals_pop
                # Update intermediary model variables if necessary
                model.update_MCMC_toolbox([self.name], realizations)


    def _sample_individual_realizations(self, data, model, realizations,temperature_inv):
        # Compute the attachment and regularity

        previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        previous_regularity = model.compute_regularity_variable(realizations[self.name]).sum(dim=1).reshape(data.n_individuals)

        # Keep previous realizations and sample new ones
        previous_reals= realizations[self.name].tensor_realizations.clone()
        realizations[self.name].tensor_realizations = self._proposal(realizations[self.name].tensor_realizations)
        # Compute the attachment and regularity

        new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        new_regularity = model.compute_regularity_variable(realizations[self.name]).sum(dim=1).reshape(data.n_individuals)

        alpha = torch.exp(-((new_regularity - previous_regularity) * temperature_inv +
                    (new_attachment - previous_attachment)))

        accepted = self._group_metropolis_step(alpha)
        self._update_acceptation_rate(list(accepted.detach().numpy()))
        self._update_std(list(accepted.detach().numpy()))
        ##### PEUT ETRE PB DE SHAPE
        accepted = accepted.unsqueeze(1)
        realizations[self.name].tensor_realizations = accepted*realizations[self.name].tensor_realizations+(1-accepted)*previous_reals

