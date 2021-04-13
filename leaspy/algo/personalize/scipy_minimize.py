from pprint import pformat

import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize

from leaspy.io.outputs.individual_parameters import IndividualParameters
from .abstract_personalize_algo import AbstractPersonalizeAlgo


class ScipyMinimize(AbstractPersonalizeAlgo):
    """
    Gradient descent based algorithm to compute individual parameters, `i.e.` personalize a model to a
    given set of subjects.
    """

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

        # logging function for convergence warnings
        # (patient_id: str, scipy_optiminize_result_dict) -> None
        if hasattr(settings, 'logger'):
            self.logger = settings.logger
        else:
            self.logger = lambda pat_id, res_dict: \
                print(f"\n<!> {pat_id}:\n\n{pformat(res_dict, indent=1)}\n")

    def _initialize_parameters(self, model):
        """
        Initialize individual parameters of one patient with group average parameter.

        ``x = [xi_mean/xi_std, tau_mean/tau_std] (+ [0.] * n_sources if multivariate model)``

        Parameters
        ----------
        model : :class:`.AbstractModel`

        Returns
        -------
        list [float]
            The individual **standardized** parameters to start with.
        """
        # rescale parameters to their natural scale so they are comparable (as well as their gradient)
        x = [model.parameters["xi_mean"] / model.parameters["xi_std"],
             model.parameters["tau_mean"] / model.parameters["tau_std"]
            ]
        if model.name != "univariate":
            x += [torch.tensor(0., dtype=torch.float32)
                  for _ in range(model.source_dimension)]
        return x

    def _pull_individual_parameters(self, x, model):
        """
        Get individual parameters as a dict[param_name: str, :class:`torch.Tensor` [1,n_dims_param]]
        from a condensed array-like version of it

        (based on the conventional order defined in :meth:`._initialize_parameters`)
        """
        tensorized_params = torch.tensor(x, dtype=torch.float32).view((1,-1)) # 1 individual

        # <!> order + rescaling of parameters
        individual_parameters = {
            'xi': tensorized_params[:,[0]] * model.parameters['xi_std'],
            'tau': tensorized_params[:,[1]] * model.parameters['tau_std'],
        }
        if 'univariate' not in model.name and model.source_dimension > 0:
            individual_parameters['sources'] = tensorized_params[:, 2:] * model.parameters['sources_std']

        return individual_parameters

    def _get_normalized_grad_tensor_from_grad_dict(self, dict_grad_tensors, model):
        """
        From a dict of gradient tensors per param (without normalization),
        returns the full tensor of gradients (= for all params, consecutively):
            * concatenated with conventional order of x0
            * normalized because we derive w.r.t. "standardized" parameter (adimensional gradient)
        """
        to_cat = [
            dict_grad_tensors['xi'] * model.parameters['xi_std'],
            dict_grad_tensors['tau'] * model.parameters['tau_std']
        ]
        if 'univariate' not in model.name and model.source_dimension > 0:
            to_cat.append( dict_grad_tensors['sources'] * model.parameters['sources_std'] )

        return torch.cat(to_cat, dim=-1).squeeze(0) # 1 individual at a time

    def _get_reconstruction_error(self, model, times, values, individual_parameters):
        """
        Compute model values minus real values of a patient for a given model, timepoints, real values & individual parameters.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        times : :class:`torch.Tensor` [n_tpts]
            Contains the individual ages corresponding to the given ``values``.
        values : :class:`torch.Tensor` [n_tpts,n_fts]
            Contains the individual true scores corresponding to the given ``times``.
        individual_parameters : dict[str, :class:`torch.Tensor` [1,n_dims_param]]
            Individual parameters as a dict

        Returns
        -------
        :class:`torch.Tensor` [n_tpts,n_fts]
            Model values minus real values.
        """

        # computation for 1 individual (level dropped after calculuus)
        predicted = model.compute_individual_tensorized(times.unsqueeze(0), individual_parameters).squeeze(0)

        return predicted - values

    def _get_regularity(self, model, individual_parameters):
        """
        Compute the regularity of a patient given his individual parameters for a given model.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.

        individual_parameters : dict[str, :class:`torch.Tensor` [n_ind,n_dims_param]]
            Individual parameters as a dict

        Returns
        -------
        regularity : :class:`torch.Tensor` [n_individuals]
            Regularity of the patient(s) corresponding to the given individual parameters.
            (Sum on all parameters)

        regularity_grads : dict[param_name: str, :class:`torch.Tensor` [n_individuals, n_dims_param]]
            Gradient of regularity term with respect to individual parameters.

        """

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
        x: array-like [float]
            Individual **standardized** parameters
            At initialization ``x = [xi_mean/xi_std, tau_mean/tau_std] (+ [0.] * n_sources if multivariate model)``

        args:
            * model : :class:`.AbstractModel`
                Model used to compute the group average parameters.
            * timepoints : :class:`torch.Tensor` [1,n_tpts]
                Contains the individual ages corresponding to the given ``values``
            * values : :class:`torch.Tensor` [n_tpts, n_fts]
                Contains the individual true scores corresponding to the given ``times``.
            * with_gradient: bool
                If True: return (objective, gradient_objective)
                Else: simply return objective

        Returns
        -------
        objective: float
            Value of the loss function (opposite of log-likelihood).

        if with_gradient is True:
            2-tuple (as expected by :func:`scipy.optimize.minimize` when ``jac=True``)
                * objective: float
                * gradient: array-like[float] of length n_dims_params
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
            grads = self._get_normalized_grad_tensor_from_grad_dict(grads, model)

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
            raise NotImplementedError(f"Algorithm loss {self.loss} is currently not implemented...")

        ## Regularity term
        regularity, regularity_grads = self._get_regularity(model, individual_parameters)

        res['objective'] += regularity.squeeze(0)

        if with_gradient:
            # add regularity term, shape (n_dims_params, )
            res['gradient'] += self._get_normalized_grad_tensor_from_grad_dict(regularity_grads, model)

            # result tuple (objective, jacobian)
            return (res['objective'].item(), res['gradient'].detach())

        else:
            # result is objective only
            return res['objective'].item()

    def _get_individual_parameters_patient(self, model, times, values, *, patient_id=None):
        """
        Compute the individual parameter by minimizing the objective loss function with scipy solver.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        times : :class:`torch.Tensor` [n_tpts]
            Contains the individual ages corresponding to the given ``values``.
        values : :class:`torch.Tensor` [n_tpts, n_fts]
            Contains the individual true scores corresponding to the given ``times``.
        patient_id : str (or None)
            ID of patient (essentially here for logging purposes when no convergence)

        Returns
        -------
        individual parameters : dict[str, :class:`torch.Tensor` [1,n_dims_param]]
            Individual parameters as a dict of tensors.
        reconstruction error : :class:`torch.Tensor` [n_tpts, n_features]
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

        individual_params_f = self._pull_individual_parameters(res.x, model)
        err_f = self._get_reconstruction_error(model, times, values, individual_params_f)

        if not res.success:
            if not self.algo_parameters['use_jacobian']:
                res['reconstruction_mae'] = round(err_f.abs().mean().item(),5) # all tpts & fts instead of mean?
                res['individual_parameters'] = individual_params_f
                #del res['x']

                self.logger(patient_id, res)

        return individual_params_f, err_f

    def _get_individual_parameters_patient_master(self, it, data, model, *, patient_id=None):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        it: int
            The iteration number.
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        data : :class:`.Dataset`
            Contains the individual scores.

        Returns
        -------
        :class:`.IndividualParameters`
            Contains the individual parameters of all patients.
        """
        times = data.get_times_patient(it)  # torch.Tensor[n_tpts]
        values = data.get_values_patient(it)  # torch.Tensor[n_tpts, n_fts]

        individual_params_tensorized, err = self._get_individual_parameters_patient(model, times, values, patient_id=patient_id)

        if self.algo_parameters.get('progress_bar', True):
            self.display_progress_bar(it, data.n_individuals, suffix='subjects')

        #return {k: v for k, v in zip(p_names, ind_patient)}
        #return individual_params_tensorized

        # transformation is needed because of IndividualParameters expectations...
        return {k: v.item() if k != 'sources' else v.detach().squeeze(0).tolist() for k,v in individual_params_tensorized.items()}

    def _get_individual_parameters(self, model, data):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        data : :class:`.Dataset` class object
            Contains the individual scores.

        Returns
        -------
        :class:`.IndividualParameters`
            Contains the individual parameters of all patients.
        """

        individual_parameters = IndividualParameters()

        #p_names = model.get_individual_variable_name()

        if self.algo_parameters.get('progress_bar', True):
            self.display_progress_bar(-1, data.n_individuals, suffix='subjects')

        ind_p_all = Parallel(n_jobs=self.algo_parameters['n_jobs'])(
            delayed(self._get_individual_parameters_patient_master)(it_pat, data, model, patient_id=id_pat)
            for it_pat, id_pat in enumerate(data.indices))

        for it_pat, ind_params_pat in enumerate(ind_p_all):
            id_pat = data.indices[it_pat]
            individual_parameters.add_individual_parameters(str(id_pat), ind_params_pat)

        return individual_parameters

