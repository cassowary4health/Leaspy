from scipy.optimize import minimize
import torch

from .abstract_personalize_algo import AbstractPersonalizeAlgo


class ScipyMinimize(AbstractPersonalizeAlgo):

    def _get_model_name(self, name):
        """
        Set name attribute

        Parameters
        ----------
        name: str
            Model's name
        """
        self.model_name = name

    def _initialize_parameters(self, model):
        """
        Initialize individual parameters with group average parameter

        Parameters
        ----------
        model: leaspy model class object

        Returns
        -------
        list of float
            By default x = [xi_mean, tau_mean] (+ [0.] * nber_of_sources if multivariate model)
        """

        x = [model.parameters["xi_mean"], model.parameters["tau_mean"]]
        if self.model_name != "univariate":
            x += [0 for _ in range(model.source_dimension)]
        return x

    def _get_attachement(self, model, times, values, x):
        """
        Compute model values minus real values for a given model, timepoints, real values & individual parameters

        Parameters
        ----------
        model: Leaspy model class object
        times: torch tensor
        values: torch tensor
        x: list of float - individual parameters

        Returns
        -------
        torch tensor - model values minus real values
        """

        xi = torch.tensor([x[0]], dtype=torch.float32).unsqueeze(0)
        tau = torch.tensor([x[1]], dtype=torch.float32).unsqueeze(0)

        if self.model_name == 'univariate':
            individual_parameters = {'xi': xi, 'tau': tau}
            err = model.compute_individual_tensorized(times, individual_parameters) - values
        else:
            sources = torch.tensor(x[2:], dtype=torch.float32).unsqueeze(0)
            individual_parameters = {'xi': xi, 'tau': tau, 'sources': sources}
            err = model.compute_individual_tensorized(times, individual_parameters) - values
        return err

    def _get_regularity(self, model, x):
        """
        Compute the regularity of the patient for a given model & individual parameters

        :param model: Leaspy model class object
        :param x: list of float - individual parameters
        :return: torch tensor - regularity of the patient corresponding to the given individual parameters
        """

        xi = torch.tensor(x[0], dtype=torch.float32)
        tau = torch.tensor(x[1], dtype=torch.float32)
        if self.model_name == 'univariate':
            iterates = zip(['xi', 'tau'], (xi, tau))
        else:
            sources = torch.tensor(x[2:], dtype=torch.float32)
            iterates = zip(['xi', 'tau', 'sources'], (xi, tau, sources))

        regularity = 0
        for key, value in iterates:
            mean = model.parameters["{0}_mean".format(key)]
            std = model.parameters["{0}_std".format(key)]
            regularity += torch.sum(model.compute_regularity_variable(value, mean, std))
        return regularity

    def _get_individual_parameters_patient(self, model, times, values):
        """
        Compute the individual parameter by minimizing the objective loss function with scipy solver

        :param model: Leaspy model class object
        :param times: torch tensor
        :param values: torch tensor
        :return: tuples - ((tau, xi, sources), error)
            tau - float
            xi - float
            sources - list of lfoat
            error - torch tensor
        """

        timepoints = times.reshape(1, -1)
        self._get_model_name(model.name)

        def obj(x, *args):
            """
            Objective loss function to minimize in order to get patient's individual parameters

            :param x: list of float - initialization of individual parameters
                By default x = [xi_mean, tau_mean] (+ [0.] * nber_of_sources if multivariate model)
            :param args: (model, timepoints, values)
                model = leaspy model class object
                timepoints = torch tensor
                values = torch tensor
            :return: float - value of the loss function
            """

            # Parameters
            model, times, values = args

            # Attachement
            xi = torch.tensor([x[0]], dtype=torch.float32).unsqueeze(0)
            tau = torch.tensor([x[1]], dtype=torch.float32).unsqueeze(0)

            if self.model_name == 'univariate':
                individual_parameters = {'xi': xi, 'tau': tau}
                attachement = model.compute_individual_tensorized(times, individual_parameters) - values
                iterates = zip(['xi', 'tau'], (xi, tau))
            else:
                sources = torch.tensor(x[2:], dtype=torch.float32).unsqueeze(0)
                individual_parameters = {'xi': xi, 'tau': tau, 'sources': sources}
                attachement = model.compute_individual_tensorized(times, individual_parameters) - values
                iterates = zip(['xi', 'tau', 'sources'], (xi, tau, sources))

            attachement = torch.sum(attachement ** 2) / (2. * model.parameters['noise_std'] ** 2)

            # Regularity
            regularity = 0
            for key, value in iterates:
                mean = model.parameters["{0}_mean".format(key)]
                std = model.parameters["{0}_std".format(key)]
                regularity += torch.sum(model.compute_regularity_variable(value, mean, std))

            return (regularity + attachement).detach().tolist()

        initial_value = self._initialize_parameters(model)
        res = minimize(obj,
                       x0=initial_value,
                       args=(model, timepoints, values),
                       method="Powell"
                       )

        if res.success is not True:
            print(res.success, res)

        xi_f, tau_f, sources_f = res.x[0], res.x[1], res.x[2:]
        err_f = self._get_attachement(model, times.unsqueeze(0), values, res.x)
        return (tau_f, xi_f, sources_f), err_f  # TODO depends on the order

    def _get_individual_parameters(self, model, data):
        """
        Compute individal parameters of all patients given a leaspy model & a leaspy dataset

        :param model: leaspy model class object
        :param data: leaspy.inputs.data.dataset class object
        :return: dict - exemple {'xi': <list of float>, 'tau': <list of float>, 'sources': <list of list of float>}
        """

        individual_parameters = {}
        for j, name_variable in enumerate(model.get_individual_variable_name()):
            individual_parameters[name_variable] = []

        # total_error = []
        total_error = torch.empty((data.n_visits, data.dimension))
        total_error_index = 0

        for idx in range(data.n_individuals):
            times = data.get_times_patient(idx)  # torch tensor
            values = data.get_values_patient(idx)  # torch tensor

            ind_patient, err = self._get_individual_parameters_patient(model, times, values)

            for j, name_variable in enumerate(model.get_individual_variable_name()):
                individual_parameters[name_variable].append(torch.tensor([ind_patient[j]], dtype=torch.float32))

            total_error[total_error_index:(total_error_index + err.squeeze(0).shape[0])] = err.squeeze(0).detach()
            total_error_index += err.squeeze(0).shape[0]

        # Print noise level
        print(total_error.std().tolist())

        infos = model.random_variable_informations()
        ## TODO change for cleaner shape update

        out = dict.fromkeys(model.get_individual_variable_name())
        for variable_ind in model.get_individual_variable_name():
            out[variable_ind] = torch.stack(individual_parameters[variable_ind]).reshape(
                shape=(data.n_individuals, infos[variable_ind]['shape'][0]))

        return out
