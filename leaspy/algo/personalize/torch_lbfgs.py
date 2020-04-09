import torch
from torch.optim import LBFGS

from .scipy_minimize import ScipyMinimize


class TorchLBFGS(ScipyMinimize):

    def __init__(self, settings):
        super(TorchLBFGS, self).__init__(settings)

        self.model_name = None
        self.initial_parameters = None
        self.minimize_kwargs = {
            'method': "Powell",
            'options': {
                'xtol': 1e-4,
                'ftol': 1e-4
            },
            # 'tol': 1e-6
        }

    def get_initial_parameters(self, idx):
        """
        Initialize individual parameters for the given subject.

        Parameters
        ----------
        idx: int
            Subject's identifier within the `Dataset` object.

        Returns
        -------
        torch.Tensor, shape = (n_individual_parameters,)
        """
        if self.initialization_method == "last_realisations":
            return self.initial_parameters[:, idx]
        else:
            return self.initial_parameters

    def objective_function(self, x, *args):
        """
        Objective loss function to minimize in order to get the patient's individual parameters.

        Parameters
        ----------
        x: torch.Tensor
            Initialization of individual parameters - in the following order (xi, tau, sources).
        args: tuple(model, timepoints, values)
            - model: leaspy model class object
                Model used to compute the group average parameters.
            - timepoints: torch.Tensor
                Contains the individual ages corresponding to the given ``values``
            - values: torch.Tensor
                Contains the individual true scores corresponding to the given ``times``.

        Returns
        -------
        objective: torch.Tensor
            Value of the loss function.
        """
        # ------ Get the additional parameters
        model, times, values = args

        # ------ Get the subject's parameters
        individual_parameters = {'tau': x[0].view(1, 1), 'xi': x[1].view(1, 1)}
        if self.model_name != 'univariate':
            individual_parameters['sources'] = x[2:].view(1, -1)
        # Parameters must be in this order: 'tau', 'xi' then 'sources'

        # ------ Compute the subject's attachment
        attachment = self._get_attachment(model, times, values, individual_parameters)
        attachment[attachment != attachment] = 0.  # Set nan to zero, not to count in the sum
        attachment = torch.sum(attachment ** 2) / (2. * model.parameters['noise_std'] ** 2)
        attachment *= self.algo_parameters['attachment_weight']

        # ------ Compute the subject's regularity
        regularity = self._get_regularity(model, individual_parameters) * self.algo_parameters['regularity_weight']

        return regularity + attachment

    def _get_individual_parameters_patient(self, model, times, values, initial_value):
        """
        Compute the individual parameters of a given patient by minimizing the objective loss function with
        scipy solver.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        times: torch.Tensor
            Contains the individual ages corresponding to the given ``values``.
        values: torch.Tensor
            Contains the individual true scores corresponding to the given ``times``.
        initial_value: torch.Tensor, shape = (n_individual_parameters,)
            Contains the initial guess fo the subject's individual parameters.

        Returns
        -------
        dict [str, torch.Tensor]
            Contains the subject's time-shift, log-acceleration & space-shifts.
        """
        timepoints = times.reshape(1, -1)
        x = initial_value.clone().detach().requires_grad_(True)

        optimizer = LBFGS([x], line_search_fn='strong_wolfe', max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = self.objective_function(x, model, timepoints, values)
            loss.backward()
            return loss
        optimizer.step(closure)

        res = optimizer.step(closure)

        return {'tau': torch.tensor(res[0], dtype=torch.float32),
                'xi': torch.tensor(res[1], dtype=torch.float32),
                'sources': torch.tensor(res[2:], dtype=torch.float32)}