import torch
from src.algo.abstract_mcmc import AbstractMCMC


class TensorMCMCSAEM(AbstractMCMC):

    def __init__(self, settings):
        super().__init__(settings)

    def _sample_population_realizations(self, data, model, realizations):

        for key in realizations.reals_pop_variable_names:
            shape_current_variable = realizations[key].shape

            # For all the dimensions
            for dim_1 in range(shape_current_variable[0]):
                for dim_2 in range(shape_current_variable[1]):

                    # Compute the attachment and regularity
                    previous_attachment = model.compute_individual_attachment_tensorized(data, realizations).sum()
                    previous_regularity = model.compute_regularity_variable(realizations[key])

                    # Keep previous realizations and sample new ones
                    previous_reals_pop = realizations[key].tensor_realizations.clone()
                    realizations[key].set_tensor_realizations_element(realizations[key].tensor_realizations[dim_1, dim_2] + self.samplers[key].sample(), (dim_1, dim_2))

                    # Update intermediary model variables if necessary
                    model.update_MCMC_toolbox([key], realizations)

                    # Compute the attachment and regularity
                    new_attachment = model.compute_individual_attachment_tensorized(data, realizations).sum()
                    new_regularity = model.compute_regularity_variable(realizations[key])

                    accepted = self._metropolisacceptation_step(new_regularity.sum(), previous_regularity.sum(),
                                                new_attachment, previous_attachment,
                                                key)

                    # Revert if not accepted
                    if not accepted:
                        # Revert realizations
                        realizations[key].tensor_realizations = previous_reals_pop
                        # Update intermediary model variables if necessary
                        model.update_MCMC_toolbox([key], realizations)


    def _sample_individual_realizations(self, data, model, realizations):


        for key_ind in realizations.reals_ind_variable_names:

            # Compute the attachment and regularity
            previous_individual_attachment = model.compute_individual_attachment_tensorized(data, realizations)
            previous_individual_regularity = model.compute_regularity_variable(realizations[key_ind])

            # Keep previous realizations and sample new ones
            previous_array_ind = realizations[key_ind].tensor_realizations
            realizations[key_ind].tensor_realizations = realizations[key_ind].tensor_realizations + self.samplers[key_ind].sample(
                shape=realizations[key_ind].tensor_realizations.shape)

            # Compute the attachment and regularity
            new_individual_attachment = model.compute_individual_attachment_tensorized(data, realizations)
            new_individual_regularity = model.compute_regularity_variable(realizations[key_ind])

            # Compute acceptation
            alpha = torch.exp(-((new_individual_attachment-previous_individual_attachment)+
                        self.temperature_inv*(new_individual_regularity- previous_individual_regularity).sum(dim=2).reshape(data.n_individuals)))
            #print(key_ind, 'Attachement {} vs {} Regularity'.format(torch.mean(pre)))
            for i, acceptation_patient in enumerate(alpha):
                accepted = self.samplers[key_ind].acceptation(acceptation_patient.detach().numpy())
                if not accepted:
                    # Update the realizations
                    realizations[key_ind].tensor_realizations[i] = previous_array_ind[i]