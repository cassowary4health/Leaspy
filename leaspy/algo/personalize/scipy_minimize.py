import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .abstract_personalize_algo import AbstractPersonalizeAlgo
from ...io.outputs.individual_parameters import IndividualParameters


class ScipyMinimize(AbstractPersonalizeAlgo):

    def __init__(self, settings):
        super().__init__(settings)

        self.minimize_kwargs = {
            'method': "Powell",
            'options': {
                'xtol': 1e-4,
                'ftol': 1e-4
            },
            # 'tol': 1e-6
        }

        if self.algo_parameters['use_jacobian']:
            self.minimize_kwargs = {
                'method': "BFGS",
                'options': {
                    'gtol': 0.01,
                    'maxiter': 200,
                },
                'tol': 5e-5
            }

    def _initialize_parameters(self, model):
        """
        Initialize individual parameters of one patient with group average parameter.

        Parameters
        ----------
        model: leaspy model class object

        Returns
        -------
        x: `list` [`float`]
            The individual parameters.
            By default x = [xi_mean, tau_mean] (+ [0.] * nber_of_sources if multivariate model)
        """
        x = [model.parameters["xi_mean"], model.parameters["tau_mean"]]
        if model.name != "univariate":
            x += [torch.tensor(0.) for _ in range(model.source_dimension)]
        return x

    def _pull_individual_parameters(self, x, model):
        """
        Get individual parameters as a dict[param_name: str, torch.Tensor [1,n_dims_param]]
        from a condensed array-like version of it
        (based on the conventional order defined in `_initialize_parameters`)
        """
        tensorized_params = torch.tensor(x, dtype=torch.float32).view((1,-1)) # 1 individual

        # <!> order
        individual_parameters = {
            'xi': tensorized_params[:,[0]],
            'tau': tensorized_params[:,[1]],
        }
        if model.name != 'univariate' and model.source_dimension > 0:
            individual_parameters['sources'] = tensorized_params[:, 2:]

        return individual_parameters

    def _get_ordered_tensor_from_dict_tensor_per_param(self, dict_tensors, model):
        """
        From a dict of tensors (per param), with param_dims being last dimension
        Return a tensor of grads for all params, concatenated with conventional order of x0.
        """
        to_cat = [dict_tensors['xi'], dict_tensors['tau']]
        if model.name != 'univariate' and model.source_dimension > 0:
            to_cat.append( dict_tensors['sources'] )

        return torch.cat(to_cat, dim=-1).squeeze(0) # 1 individual at a time

    def _get_reconstruction_error(self, model, times, values, x):
        """
        Compute model values minus real values of a patient for a given model, timepoints, real values &
        individual parameters.

        Parameters
        ----------
        model: Leaspy model class object
            Model used to compute the group average parameters.
        times: `torch.Tensor` [n_tpts]
            Contains the individual ages corresponding to the given ``values``.
        values: `torch.Tensor` [n_tpts,n_fts]
            Contains the individual true scores corresponding to the given ``times``.
        x: `list` [`float`]
            The individual parameters.

        Returns
        -------
        err: `torch.Tensor` [n_tpts,n_fts]
            Model values minus real values.
        """
        individual_parameters = self._pull_individual_parameters(x, model)
        # computation for 1 individual (level dropped after calculuus)
        predicted = model.compute_individual_tensorized(times.unsqueeze(0), individual_parameters).squeeze(0)

        return predicted - values

    def _get_regularity(self, model, individual_parameters):
        """
        Compute the regularity of a patient given his individual parameters for a given model.

        Parameters
        ----------
        model: Leaspy model class object
            Model used to compute the group average parameters.

        individual_parameters: dict[str, torch.Tensor[n_ind,n_dims_param]]
            Individual parameters as a dict

        Returns
        -------
        2-tuple:

        - regularity: `torch.Tensor` [n_individuals]
            Regularity of the patient(s) corresponding to the given individual parameters.
            (Sum on all parameters)

        - regularity_grads: dict[param_name: str, `torch.Tensor` [n_individuals, n_dims_param]]
            Gradient of regularity term with respect to individual parameters.

        """
        #individual_parameters = self._pull_individual_parameters(x, model)

        regularity = 0
        regularity_grads = {}

        for param_name, param_val in individual_parameters.items():
            # priors on this parameter
            priors = dict(
                mean = model.parameters[param_name+"_mean"],
                std = model.parameters[param_name+"_std"]
            )

            # summation term
            regularity += model.compute_regularity_variable(param_val, **priors).sum(dim=1)

            # derivatives: formula below is for Normal parameters priors only
            # TODO? create a more generic method in model `compute_regularity_variable_gradient`? but to do so we should probably wait to have some more generic `compute_regularity_variable` as well (at least pass the parameter name to this method to compute regularity term)
            regularity_grads[param_name] = (param_val - priors['mean']) / (priors['std']**2)

        return (regularity, regularity_grads)


    def obj(self, x, *args):
        """
        Objective loss function to minimize in order to get patient's individual parameters

        Parameters
        ----------
        x: array-like [`float`]
            Initialization of individual parameters
            By default x = [xi_mean, tau_mean] (+ [0.] * nber_of_sources if multivariate model)

        args:
            - model: leaspy model class object
                Model used to compute the group average parameters.
            - timepoints: `torch.Tensor`[1,n_tpts]
                Contains the individual ages corresponding to the given ``values``
            - values: `torch.Tensor`[n_tpts, n_fts]
                Contains the individual true scores corresponding to the given ``times``.
            - with_gradient boolean
                If True: return (objective, gradient_objective)
                Else: simply return objective

        Returns
        -------
        objective: `float`
            Value of the loss function (opposite of log-likelihood).

        if with_gradient is True:
            2-tuple (as expected by scipy.optimize.minimize when jac=True)
            - objective: float
            - gradient: array-like[float] of length n_dims_params
        """

        # Extra arguments passed by scipy minimize
        model, times, values, with_gradient = args

        ## Attachment term
        individual_parameters = self._pull_individual_parameters(x, model)

        # compute 1 individual at a time (1st dimension is squeezed)
        predicted = model.compute_individual_tensorized(times, individual_parameters).squeeze(0)
        diff = predicted - values # tensor j,k (j=visits, k=features)
        nans = torch.isnan(diff)
        diff[nans] = 0.  # set nans to zero, not to count in the sum

        # compute  gradient of model with respect to individual parameters
        if with_gradient:
            grads = model.compute_jacobian_tensorized(times, individual_parameters)
            # put derivatives consecutively in the right order: shape [n_tpts,n_fts,n_dims_params]
            grads = self._get_ordered_tensor_from_dict_tensor_per_param(grads, model)

        # Placeholder for result (objective and, if needed, gradient)
        res = {}

        # Different losses implemented
        if 'MSE' in self.loss:
            noise_var = model.parameters['noise_std'] * model.parameters['noise_std']
            noise_var = noise_var.expand((1, model.dimension)) # tensor 1,n_fts (works with diagonal noise or scalar noise)
            res['objective'] = torch.sum((0.5 / noise_var) @ (diff * diff).t()) # <!> noise per feature

            if with_gradient:
                res['gradient'] = torch.sum((diff / noise_var).unsqueeze(-1) * grads, dim=(0,1))

        elif self.loss == 'crossentropy':
            predicted = torch.clamp(predicted, 1e-38, 1. - 1e-7)  # safety before taking the log # @P-E: why clamping not symmetric?
            neg_crossentropy = values * torch.log(predicted) + (1. - values) * torch.log(1. - predicted)
            neg_crossentropy[nans] = 0. # set nans to zero, not to count in the sum
            res['objective'] = -torch.sum(neg_crossentropy)

            if with_gradient:
                crossentropy_fact = diff / (predicted * (1. - predicted))
                res['gradient'] = torch.sum(crossentropy_fact.unsqueeze(-1) * grads, dim=(0,1))

        else:
            raise NotImplementedError

        ## Regularity term
        regularity, regularity_grads = self._get_regularity(model, individual_parameters)

        res['objective'] += regularity.squeeze(0)

        if with_gradient:
            # add regularity term, shape (n_dims_params, )
            res['gradient'] += self._get_ordered_tensor_from_dict_tensor_per_param(regularity_grads, model)

            # result tuple (objective, jacobian)
            return (res['objective'].item(), res['gradient'].detach())

        else:
            # result is objective only
            return res['objective'].item()


    def _get_individual_parameters_patient(self, model, times, values):
        """
        Compute the individual parameter by minimizing the objective loss function with scipy solver.

        Parameters
        ----------
        model: Leaspy model class object
            Model used to compute the group average parameters.
        times: `torch.Tensor` [n_tpts]
            Contains the individual ages corresponding to the given ``values``.
        values: `torch.Tensor` [n_tpts, n_fts]
            Contains the individual true scores corresponding to the given ``times``.

        Returns
        -------
            - tau - `float`
                Individual time-shift.
            - xi - `float`
                Individual log-acceleration.
            - sources - `list` [`float`]
                Individual space-shifts.
            - error - `torch.Tensor`
                Model values minus real values.
        """

        # optimize by sending exact gradient of optimized function?
        with_jac = self.algo_parameters.get('use_jacobian', False)

        initial_value = self._initialize_parameters(model)
        res = minimize(self.obj,
                       jac=with_jac,
                       x0=initial_value,
                       args=(model, times.unsqueeze(0), values, with_jac),
                       **self.minimize_kwargs
                       )

        if res.success is not True:
            print(res.success, res)

        xi_f, tau_f, sources_f = res.x[0], res.x[1], res.x[2:] # TODO? _pull_individual_parameters here instead?
        err_f = self._get_reconstruction_error(model, times, values, res.x)

        return (tau_f, xi_f, sources_f), err_f  # TODO depends on the order

    def _get_individual_parameters_patient_master(self, it, data, model, p_names):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        it: int
            The iteration number.
        model: leaspy model class object
            Model used to compute the group average parameters.
        data: leaspy.io.data.dataset.Dataset class object
            Contains the individual scores.
        p_names: list of str
            Contains the individual parameters' names.

        Returns
        -------
        leaspy.io.outputs.individual_parameters.IndividualParameters
            Contains the individual parameters of all patients.
        """
        times = data.get_times_patient(it)  # torch.Tensor[n_tpts]
        values = data.get_values_patient(it)  # torch.Tensor[n_tpts, n_fts]

        ind_patient, err = self._get_individual_parameters_patient(model, times, values)

        if self.algo_parameters.get('progress_bar',True):
            self.display_progress_bar(it, data.n_individuals, suffix='subjects')

        return {k: v for k, v in zip(p_names, ind_patient)}

    def _get_individual_parameters(self, model, data):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model: leaspy model class object
            Model used to compute the group average parameters.
        data: leaspy.io.data.dataset.Dataset class object
            Contains the individual scores.

        Returns
        -------
        leaspy.io.outputs.individual_parameters.IndividualParameters
            Contains the individual parameters of all patients.
        """

        individual_parameters = IndividualParameters()

        p_names = model.get_individual_variable_name()

        if self.algo_parameters.get('progress_bar',True):
            self.display_progress_bar(-1, data.n_individuals, suffix='subjects')

        ind_p_all = Parallel(n_jobs=self.algo_parameters['n_jobs'])(
            delayed(self._get_individual_parameters_patient_master)(it, data, model, p_names) for it in range(data.n_individuals))

        for it, ind_p in enumerate(ind_p_all):
            idx = data.indices[it]
            individual_parameters.add_individual_parameters(str(idx), ind_p)

        return individual_parameters
