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

    @staticmethod
    def _initialize_parameters(model):
        """
        Initialize individual parameters of one patient with group average parameter. Used as initial value for
        scipy ``minimize`` function. Then, `_initialize_parameters` must return an

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.

        Returns
        -------
        x: list [float]
            The individual parameters.
        """
        length = 2
        if model.name != "univariate":
            length += model.source_dimension
        x = [0.] * length
        x[0] = model.parameters["xi_mean"].item()
        x[1] = model.parameters["tau_mean"].item()
        return x

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
            The individual parameters.

        Returns
        -------
        torch.Tensor
            Model values minus real values.
        """
        return model.compute_individual_tensorized(times, individual_parameters) - values

    @staticmethod
    def _get_regularity(model, individual_parameters):
        """
        Compute the regularity of a patient given his individual parameters for a given model.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        individual_parameters: dict [str, torch.Tensor]
            The individual parameters.

        Returns
        -------
        regularity: torch.Tensor
            Regularity of the patient corresponding to the given individual parameters.
        """
        regularity = 0
        for key, value in individual_parameters.items():
            mean = model.parameters["{0}_mean".format(key)]
            std = model.parameters["{0}_std".format(key)]
            regularity += torch.sum(model.compute_regularity_variable(value, mean, std))
        return regularity

    def _get_individual_parameters_patient_master(self, model, data, idx, initial_value):
        """
        From the `Dataset` object, extract the values and time-points the subject ``idx``, then call the method to
        compute the individual parameters of a given patient.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        data: leaspy.inputs.data.dataset.Dataset
            Contains the individual scores.
        idx: int
            Subject's identifier within the `Dataset` object.
        initial_value: list [float]
            Contain the initial values of the patient individual parameters. By default, they are the model average
            parameters.

        Returns
        -------
        dict [str, torch.Tensor]
            Contains the subject's time-shift, log-acceleration & space-shifts.
        """
        values = data.get_values_patient(idx)  # torch.Tensor
        timepoints = data.get_times_patient(idx)  # torch.Tensor
        return self._get_individual_parameters_patient(model, timepoints, values, initial_value)

    def _get_individual_parameters_patient(self, model, times, values, initial_value=None):
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
        initial_value: list [float]
            Contain the initial values of the patient individual parameters. By default, they are the model average
            parameters.

        Returns
        -------
        dict [str, torch.Tensor]
            Contains the subject's time-shift, log-acceleration & space-shifts.
        """
        def obj(x, *args):
            """
            Objective loss function to minimize in order to get the patient's individual parameters.

            Parameters
            ----------
            x: numpy.ndarray
                Initialization of individual parameters.
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
            # Parameters
            model, times, values = args

            # Individual parameters
            xi = torch.tensor([[x[0]]], dtype=torch.float32)
            tau = torch.tensor([[x[1]]], dtype=torch.float32)
            individual_parameters = {'xi': xi, 'tau': tau}
            if self.model_name != 'univariate':
                individual_parameters['sources'] = torch.tensor([x[2:]], dtype=torch.float32)

            # Attachment
            attachment = self._get_attachment(model, times, values, individual_parameters)
            attachment[attachment != attachment] = 0.  # Set nan to zero, not to count in the sum
            attachment = torch.sum(attachment ** 2) / (2. * model.parameters['noise_std'] ** 2)

            # Regularity
            regularity = self._get_regularity(model, individual_parameters)

            return (regularity + attachment).detach().item()

        timepoints = times.reshape(1, -1)
        if initial_value is None:
            initial_value = self._initialize_parameters(model)

        res = minimize(obj,
                       x0=initial_value,
                       args=(model, timepoints, values),
                       **self.minimize_kwargs
                       )

        if res.success is not True:
            print(res.success, res)

        return {'xi': torch.tensor(res.x[0], dtype=torch.float32),
                'tau': torch.tensor(res.x[1], dtype=torch.float32),
                'sources': torch.tensor(res.x[2:], dtype=torch.float32)}

    def _get_individual_parameters(self, model, data):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model: leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        data: leaspy.inputs.data.dataset.Dataset
            Contains the individual scores.

        Returns
        -------
        out: dict [str, torch.Tensor]
            Contains the individual parameters of all patients.
        """
        self._set_model_name(model.name)
        initial_value = self._initialize_parameters(model)

        individual_parameters = {}
        for name_variable in model.get_individual_variable_name():
            individual_parameters[name_variable] = []

        if self.algo_parameters['parallel']:
            # Use joblib
            individual_parameters = Parallel(n_jobs=self.algo_parameters['n_jobs'])(
                delayed(self._get_individual_parameters_patient_master)(model, data, idx, initial_value)
                for idx in range(data.n_individuals))
            out = {key: torch.tensor([ind_param[key] for ind_param in individual_parameters], dtype=torch.float32)
                   for key in model.get_individual_variable_name()}
            out['xi'] = out['xi'].view(-1, 1)
            out['tau'] = out['tau'].view(-1, 1)
            if out['sources'].ndim == 1:
                out['sources'] = out['sources'].view(-1, 1)  # if only one sources

        else:
            # Simple for loop
            for idx in range(data.n_individuals):
                ind_patient = self._get_individual_parameters_patient_master(model, data, idx, initial_value)
                for name_variable in model.get_individual_variable_name():
                    individual_parameters[name_variable].append(ind_patient[name_variable])
            infos = model.random_variable_informations()

            out = dict.fromkeys(model.get_individual_variable_name())
            for variable_ind in model.get_individual_variable_name():
                out[variable_ind] = torch.stack(individual_parameters[variable_ind]).reshape(
                    shape=(data.n_individuals, infos[variable_ind]['shape'][0]))

        return out
