import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .abstract_personalize_algo import AbstractPersonalizeAlgo


class ScipyMinimize(AbstractPersonalizeAlgo):

    def __init__(self, settings):
        """
        Constructor.

        Parameters
        ----------
        settings: leaspy.intputs.settings.algorithm_settings.AlgorithmSettings
        """
        super(ScipyMinimize, self).__init__(settings)

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

    def _set_model_name(self, name):
        """
        Set name attribute.

        Parameters
        ----------
        name: str
            Model's name.
        """
        self.model_name = name

    def _initialize_parameters(self, model):
        """
        Initialize all the individual parameters for all subjects. Used as initial value for scipy ``minimize``
        function. If the ``initialization_method`` is set to ``"last_realisations"``, the model's last realisation is
        used as initial guess. Else, the posterior mean of the individual parameters distribution is used as
        initial guess.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        """
        if self.initialization_method == "last_realisations":
            _, ind_param = model._last_realisations.to_dict()
            self.initial_parameters = torch.cat((ind_param['tau'], ind_param['xi'], ind_param['sources']), dim=1)
        else:
            self.initial_parameters = torch.zeros(2 + model.source_dimension)
            self.initial_parameters[0] = model.parameters["tau_mean"].item()
            self.initial_parameters[1] = model.parameters["xi_mean"].item()
            # self.initial_parameters[:2] = model.individual_parameters_posterior_distribution.loc[:2]

    def get_initial_parameters(self, idx):
        """
        Initialize individual parameters for the given subject.

        Parameters
        ----------
        idx: int
            Subject's identifier within the `Dataset` object.

        Returns
        -------
        numpy.ndarray, shape = (n_individual_parameters,)
        """
        if self.initialization_method == "last_realisations":
            return self.initial_parameters[:, idx].numpy()
        else:
            return self.initial_parameters.numpy()

    @staticmethod
    def _get_attachment(model, times, values, individual_parameters):
        """
        Compute model values minus real values of a patient for a given model, timepoints, real values &
        individual parameters.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        times: torch.Tensor
            Contains the individual ages corresponding to the given ``values``.
        values: torch.Tensor
            Contains the individual true scores corresponding to the given ``times``.
        individual_parameters: dict [str, torch.Tensor]
            The individual parameters. The sources' shape is (1, n_sources), tau and xi have shape of (1, 1).

        Returns
        -------
        torch.Tensor
            Model values minus real values.
        """
        return model.compute_individual_tensorized(times, individual_parameters) - values

    def _get_regularity(self, model, individual_parameters):
        """
        Compute the regularity of a patient given his individual parameters for a given model. The choice of method of
        regularization depend of the settings parameters ``"regularity_method"``.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        individual_parameters: dict [str, torch.Tensor]
            The individual parameters. The sources' shape is (1, n_sources), tau and xi have shape of (1, 1).

        Returns
        -------
        regularity: torch.Tensor
            Regularity of the patient corresponding to the given individual parameters.
        """
        if self.algo_parameters['regularity_method'] == "prior":
            return self._get_prior_regularity(model, individual_parameters)
        elif self.algo_parameters['regularity_method'] == "posterior":
            return self._get_posterior_regularity(model, individual_parameters)
        elif self.algo_parameters['regularity_method'] == "conditional_posterior":
            return self._get_conditional_posterior_regularity(model, individual_parameters)
        else:
            raise ValueError('The parameter "regularity_method" must be "prior", "posterior" or '
                             '"conditional_posterior"! You gave {}'.format(self.algo_parameters['regularity_method']))

    @staticmethod
    def _get_prior_regularity(model, individual_parameters):
        """
        Compute the regularity of a patient given his individual parameters for a given model. In this settings,
        the individual variables are assumed to be univariate gaussians independent for each other.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        individual_parameters: dict [str, torch.Tensor]
            The individual parameters. The sources' shape is (1, n_sources), tau and xi have shape of (1, 1).

        Returns
        -------
        regularity: torch.Tensor
            Regularity of the patient corresponding to the given individual parameters.
        """
        regularity = torch.zeros(1)
        for key, value in individual_parameters.items():
            mean = model.parameters["{0}_mean".format(key)]
            std = model.parameters["{0}_std".format(key)]
            regularity += torch.sum(model.compute_regularity_variable(value, mean, std))
        return regularity

    @staticmethod
    def _get_posterior_regularity(model, individual_parameters):
        """
        Compute the regularity of a patient given his individual parameters for a given model. In this settings,
        the individual variables are assumed to be a multivariate gaussian distribution.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        individual_parameters: dict [str, torch.Tensor]
            The individual parameters. The sources' shape is (1, n_sources), tau and xi have shape of (1, 1).

        Returns
        -------
        regularity: torch.Tensor
            Regularity of the patient corresponding to the given individual parameters.
        """
        tensor_list = [individual_parameters['tau'].unsqueeze(0), individual_parameters['xi'].unsqueeze(0)]
        if model.name != 'univariate':
            tensor_list.append([individual_parameters['sources'].unsqueeze(0)])
        value = torch.cat(tensor_list, dim=-1).squeeze()
        # Reshaped into torch.Tensor([tau, xi, s0, s1, ...])

        return model.compute_multivariate_gaussian_posterior_regularity(value)

    @staticmethod
    def _get_conditional_posterior_regularity(model, individual_parameters):
        """
        Given the individual parameter of a subject, compute its regularity assuming the individual parameters
        follow a multivariate normal distribution. We use the posterior distribution of the individual
        parameters of the cohort on which the model has been calibrated. The parameter tau & i are assumed to be
        independent (from each other and from the sources). To compute the regularity of the sources, the conditional
        mean and covariance matrix of the sources knowing tau and i are computed.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        individual_parameters: dict [str, torch.Tensor]
            The individual parameters. The sources' shape is (1, n_sources), tau and xi have shape of (1, 1).

        Returns
        -------
        regularity: torch.Tensor
            Regularity of the patient corresponding to the given individual parameters.
        """
        # ------ First, compute regularity of the subject's tau & xi to their respective
        # posterior distribution assuming they are independent
        regularity = torch.tensor(0.)
        for i, param in enumerate(('tau', 'xi')):  # tau, xi in this order
            mean = model.individual_parameters_posterior_distribution.loc[i]
            std = model.individual_parameters_posterior_distribution.covariance_matrix[i, i].sqrt()
            regularity += model.compute_regularity_variable(individual_parameters[param], mean, std).sum()

        # ------ Then compute the sources' regularity and add it to the one of tau and xi
        if model.name != 'univariate':
            tau_xi = torch.tensor([individual_parameters['tau'].item(),
                                   individual_parameters['xi'].item()], dtype=torch.float32)
            sources = individual_parameters['sources'].squeeze()
            regularity += model.compute_multivariate_sources_regularity(tau_xi, sources).sum()

        return regularity

    def _get_individual_parameters_patient_master(self, model, dataset, idx):
        """
        From the `Dataset` object, extract the values and time-points the subject ``idx``, then call the method to
        compute the individual parameters of a given patient.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        dataset: leaspy.inputs.data.dataset.Dataset
            Contains the individual scores.
        idx: int
            Subject's identifier within the `Dataset` object.

        Returns
        -------
        dict [str, torch.Tensor]
            Contains the subject's time-shift, log-acceleration & space-shifts.
        """
        values = dataset.get_values_patient(idx)
        timepoints = dataset.get_times_patient(idx)
        initial_value = self.get_initial_parameters(idx)
        return self._get_individual_parameters_patient(model, timepoints, values, initial_value)

    def objective_function_bis(self, x, *args):
        """
        Objective loss function to minimize in order to get the patient's individual parameters.

        Parameters
        ----------
        x: numpy.ndarray
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
        objective: float
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

    def objective_function(self, x, *args):
        """
        Objective loss function to minimize in order to get the patient's individual parameters.

        Parameters
        ----------
        x: numpy.ndarray
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
        objective: float
            Value of the loss function.
        """
        # ------ Get the additional parameters
        model, times, values = args

        # ------ Get the subject's parameters
        individual_parameters = {'tau': torch.tensor([[x[0]]], dtype=torch.float32),
                                 'xi': torch.tensor([[x[1]]], dtype=torch.float32)}
        if self.model_name != 'univariate':
            individual_parameters['sources'] = torch.tensor([x[2:]], dtype=torch.float32)
        # Parameters must be in this order: 'tau', 'xi' then 'sources'

        # ------ Compute the subject's attachment
        attachment = self._get_attachment(model, times, values, individual_parameters)
        attachment[attachment != attachment] = 0.  # Set nan to zero, not to count in the sum
        attachment = torch.sum(attachment ** 2) / (2. * model.parameters['noise_std'] ** 2)
        attachment *= self.algo_parameters['attachment_weight']

        # ------ Compute the subject's regularity
        regularity = self._get_regularity(model, individual_parameters) * self.algo_parameters['regularity_weight']

        return (regularity + attachment).detach().item()

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
        initial_value: numpy.ndarray, shape = (n_individual_parameters,)
            Contains the initial guess fo the subject's individual parameters.

        Returns
        -------
        dict [str, torch.Tensor]
            Contains the subject's time-shift, log-acceleration & space-shifts.
        """
        timepoints = times.reshape(1, -1)

        res = minimize(self.objective_function,
                       x0=initial_value,
                       args=(model, timepoints, values),
                       **self.minimize_kwargs)

        if res.success is not True:
            print(res.success, res)

        return {'tau': torch.tensor(res.x[0], dtype=torch.float32),
                'xi': torch.tensor(res.x[1], dtype=torch.float32),
                'sources': torch.tensor(res.x[2:], dtype=torch.float32)}

    def _get_individual_parameters(self, model, dataset):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        dataset: leaspy.inputs.data.dataset.Dataset
            Contains the individual scores.

        Returns
        -------
        out: dict [str, torch.Tensor]
            Contains the individual parameters of all patients.
        """
        self._set_model_name(model.name)
        infos = model.random_variable_informations()
        self._initialize_parameters(model)

        if self.algo_parameters['parallel']:
            # Use joblib
            individual_parameters = Parallel(n_jobs=self.algo_parameters['n_jobs'])(
                delayed(self._get_individual_parameters_patient_master)(model, dataset, idx)
                for idx in range(dataset.n_individuals))
            out = {key: torch.tensor([ind_param[key] for ind_param in individual_parameters], dtype=torch.float32)
                   for key in model.get_individual_variable_name()}
            for variable_ind in model.get_individual_variable_name():
                out[variable_ind] = out[variable_ind].reshape(shape=(dataset.n_individuals,
                                                                     infos[variable_ind]['shape'][0]))
        else:
            individual_parameters = {}
            for name_variable in model.get_individual_variable_name():
                individual_parameters[name_variable] = []
            # Simple for loop
            for idx in range(dataset.n_individuals):
                ind_patient = self._get_individual_parameters_patient_master(model, dataset, idx)
                for name_variable in model.get_individual_variable_name():
                    individual_parameters[name_variable].append(ind_patient[name_variable])

            out = dict.fromkeys(model.get_individual_variable_name())
            for variable_ind in model.get_individual_variable_name():
                out[variable_ind] = torch.stack(individual_parameters[variable_ind]).reshape(
                    shape=(dataset.n_individuals, infos[variable_ind]['shape'][0]))

        return out
