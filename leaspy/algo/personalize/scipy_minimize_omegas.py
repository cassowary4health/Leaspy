import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .abstract_personalize_algo import AbstractPersonalizeAlgo


class ScipyMinimizeOmegas(AbstractPersonalizeAlgo):

    def __init__(self, settings):
        """
        Constructor.

        Parameters
        ----------
        settings: leaspy.intputs.settings.algorithm_settings.AlgorithmSettings
        """
        super(ScipyMinimizeOmegas, self).__init__(settings)

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
        self.initial_parameters = torch.zeros(2 + len(model.features))
        if model.individual_parameters_posterior_distribution is None:
            self.initial_parameters[0] = model.parameters["tau_mean"].item()
            self.initial_parameters[1] = model.parameters["xi_mean"].item()
        else:
            self.initial_parameters[:2] = model.individual_parameters_posterior_distribution.loc[:2]
            self.initial_parameters[2:] = model._omegas_posterior_mean

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
        attachment = model.compute_individual_tensorized_omegas(times, individual_parameters) - values
        attachment[attachment != attachment] = 0.  # Set nan to zero, not to count in the sum
        return torch.sum(attachment ** 2) / (2. * model.parameters['noise_std'] ** 2)

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
        if self.algo_parameters['regularity_method'] == "posterior_omegas":
            return self._get_posterior_regularity_omegas(
                model, individual_parameters, solve_method=self.algo_parameters["solve_method"])
        else:
            raise ValueError('The parameter "regularity_method" must be "posterior_omegas" ! '
                             'You gave {}'.format(self.algo_parameters['regularity_method']))

    @staticmethod
    def _get_posterior_regularity_omegas(model, individual_parameters, solve_method='numpy_lstsq'):
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
        # ------ First, compute regularity of the subject's tau & xi to their respective
        # posterior distribution assuming they are independent
        regularity = torch.tensor(0., dtype=torch.float32)
        for i, param in enumerate(('tau', 'xi')):  # tau, xi in this order
            mean = model.individual_parameters_posterior_distribution.loc[i]
            std = model.individual_parameters_posterior_distribution.covariance_matrix[i, i].sqrt()
            regularity += model.compute_regularity_variable(individual_parameters[param], mean, std).sum()

        # ------ Then compute the omegas' regularity and add it to the one of tau and xi
        if model.name != 'univariate':
            regularity += model.compute_multivariate_omegas_regularity(individual_parameters['omegas'],
                                                                       solve_method=solve_method).sum()

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
            individual_parameters['omegas'] = torch.tensor([x[2:]], dtype=torch.float32)
        # Parameters must be in this order: 'tau', 'xi' then 'sources'

        # ------ Compute the subject's attachment
        attachment = self._get_attachment(model, times, values, individual_parameters)
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

        if self.algo_parameters['verbose'] == 'debug':
            def callback(xk):
                individual_parameters = {'tau': torch.tensor([xk[0]], dtype=torch.float32),
                                         'xi': torch.tensor([xk[1]], dtype=torch.float32),
                                         'omegas': torch.tensor([xk[2:]], dtype=torch.float32)}
                attachment = self._get_attachment(model, times, values, individual_parameters)
                attachment *= self.algo_parameters['attachment_weight']
                regularity = self._get_regularity(model, individual_parameters) * self.algo_parameters['regularity_weight']
                print("Attachment : %.5f - Regularity : %.5f" % (attachment.item(), regularity.item()))
        else:
            callback = None

        res = minimize(self.objective_function,
                       x0=initial_value,
                       args=(model, timepoints, values),
                       callback=callback,
                       **self.minimize_kwargs)

        if res.success is not True:
            print(res.success, res)

        return {'tau': torch.tensor(res.x[0], dtype=torch.float32),
                'xi': torch.tensor(res.x[1], dtype=torch.float32),
                'omegas': torch.tensor(res.x[2:], dtype=torch.float32)}

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

        if self.algo_parameters['regularity_method'] == 'posterior_omegas':
            model.set_omegas_posterior_covariance_inverse(self.algo_parameters['omegas_regularity_factor'])
        if self.algo_parameters['regularity_method'] == 'posterior_omegas_bis':
            model.set_omegas_posterior_covariance_inverse_bis(self.algo_parameters['omegas_regularity_factor'])
            # TODO : raise error if model is univariate and method is "posterior_omegas"

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
            for name_variable in ('tau', 'xi', 'omegas'):
                individual_parameters[name_variable] = []
            # Simple for loop
            for idx in range(dataset.n_individuals):
                ind_patient = self._get_individual_parameters_patient_master(model, dataset, idx)
                for name_variable in ('tau', 'xi', 'omegas'):
                    individual_parameters[name_variable].append(ind_patient[name_variable])

            out = dict.fromkeys(('tau', 'xi', 'omegas'))
            for variable_ind in model.get_individual_variable_name():
                if variable_ind in ('tau', 'xi'):
                    out[variable_ind] = torch.stack(individual_parameters[variable_ind]).reshape(
                        shape=(dataset.n_individuals, infos[variable_ind]['shape'][0]))
                else:
                    out['omegas'] = torch.stack(individual_parameters['omegas']).reshape(
                        shape=(dataset.n_individuals, len(model.features)))

        return out
