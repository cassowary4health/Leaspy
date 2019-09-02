import torch
from .abstract_personalize_algo import AbstractPersonalizeAlgo


class GradientDescentPersonalize(AbstractPersonalizeAlgo):

    def __init__(self, settings):

        self.name = "Gradient Descent - Personalize"
        self.realizations = None
        self.task = None
        self.algo_parameters = settings.parameters
        self.current_iteration = 0
        self.path_output = 'output/'
        self.personalize_output = {}

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):

        # MCMC toolbox (cache variables for speed-ups + tricks)
        model.initialize_MCMC_toolbox()
        self._initialize_torchvariables(realizations)

        return realizations

    def _initialize_torchvariables(self, realizations):
        for name, realization in realizations.realizations.items():
            realization.to_torch_Variable()



    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):

        # Update intermediary model variables if necessary
        model.update_MCMC_toolbox(["all"], realizations)

        # Compute loss
        previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
        previous_regularity = 0
        for key in realizations.keys():
            previous_regularity += model.compute_regularity_realization(realizations[key]).sum()
        loss = previous_attachment + previous_regularity

        # Do backward and backprop on realizations
        loss.backward()


        # Update ind
        with torch.no_grad():
            for key in realizations.reals_ind_variable_names:
                eps =self.algo_parameters['learning_rate']
                realizations[key].tensor_realizations -= eps*realizations[key].tensor_realizations.grad
                realizations[key].tensor_realizations.grad.zero_()

        self.personalize_output["map"] = realizations

        """
        noise = (model.compute_sum_squared_tensorized(data, realizations).sum() / (
                    data.n_visits * data.dimension)).detach().numpy().tolist()
        print(noise)"""
