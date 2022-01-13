from pprint import pformat
import warnings

import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize

from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo

from leaspy.exceptions import LeaspyAlgoInputError


class ScipyMinimize(AbstractPersonalizeAlgo):
    """
    Gradient descent based algorithm to compute individual parameters,
    `i.e.` personalize a model to a given set of subjects.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.

    Attributes
    ----------
    print_convergence_issues : bool
        Should we display all convergence issues returned by `scipy.optimize`?
        By default display convergences issues iff not BFGS method
        Note that it is not used if custom `logger` is defined in settings.
    minimize_kwargs : kwargs
        Keyword arguments passed to :func:`scipy.optimize.minimize`
    """

    name = 'scipy_minimize'

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
            # <!> this custom params will remain even if falling back to without jacobian when not implemented
            self.minimize_kwargs = {
                'method': "BFGS",
                'options': {
                    'gtol': 0.01,
                    'maxiter': 200,
                },
                'tol': 5e-5
            }

        # by default display convergences issues iff not BFGS method (not used if custom logger defined)
        self.print_convergence_issues = self.minimize_kwargs['method'].upper() != 'BFGS'

        # logging function for convergence warnings
        # (patient_id: str, scipy_minize_result_dict) -> None
        if hasattr(settings, 'logger'):
            self.logger = settings.logger
        else:
            if self.print_convergence_issues:
                self.logger = lambda pat_id, res_dict: \
                    print(f"\n<!> {pat_id}:\n{pformat(res_dict, indent=1)}\n")
            else:
                self.logger = lambda *args, **kwargs: None

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

        # computation for 1 individual (level dropped after computation)
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
        x : array-like [float]
            Individual **standardized** parameters
            At initialization ``x = [xi_mean/xi_std, tau_mean/tau_std] (+ [0.] * n_sources if multivariate model)``

        *args
            * model : :class:`.AbstractModel`
                Model used to compute the group average parameters.
            * timepoints : :class:`torch.Tensor` [1,n_tpts]
                Contains the individual ages corresponding to the given ``values``
            * values : :class:`torch.Tensor` [n_tpts, n_fts]
                Contains the individual true scores corresponding to the given ``times``.
            * with_gradient : bool
                * If True: return (objective, gradient_objective)
                * Else: simply return objective

        Returns
        -------
        objective : float
            Value of the loss function (opposite of log-likelihood).

        if `with_gradient` is True:
            2-tuple (as expected by :func:`scipy.optimize.minimize` when ``jac=True``)
                * objective : float
                * gradient : array-like[float] of length n_dims_params

        Raises
        ------
        :exc:`.LeaspyAlgoInputError`
            if noise model is not currently supported by algorithm.
            TODO: everything that is not generic here concerning noise structure should be handle by model/NoiseModel directly!!!!
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

        # compute gradient of model with respect to individual parameters
        grads = None
        if with_gradient:
            grads = model.compute_jacobian_tensorized(times, individual_parameters)
            # put derivatives consecutively in the right order: shape [n_tpts,n_fts,n_dims_params]
            grads = self._get_normalized_grad_tensor_from_grad_dict(grads, model)

        # Placeholder for result (objective and, if needed, gradient)
        res = {}

        # Loss is based on log-likelihood for model, which ultimately depends on noise structure
        # TODO: should be directly handled in model or NoiseModel
        if 'gaussian' in model.noise_model:
            noise_var = model.parameters['noise_std'] * model.parameters['noise_std']
            noise_var = noise_var.expand((1, model.dimension)) # tensor 1,n_fts (works with diagonal noise or scalar noise)
            res['objective'] = torch.sum((0.5 / noise_var) @ (diff * diff).t()) # <!> noise per feature

            if with_gradient:
                res['gradient'] = torch.sum((diff / noise_var).unsqueeze(-1) * grads, dim=(0,1))

        elif model.noise_model == 'bernoulli':
            # safety before taking the log: cf. torch.finfo(torch.float32).eps ~= 1.19e-7
            predicted = torch.clamp(predicted, 1e-7, 1. - 1e-7)
            neg_crossentropy = values * torch.log(predicted) + (1. - values) * torch.log(1. - predicted)
            neg_crossentropy[nans] = 0. # set nans to zero, not to count in the sum
            res['objective'] = -torch.sum(neg_crossentropy)

            if with_gradient:
                crossentropy_fact = diff / (predicted * (1. - predicted))
                res['gradient'] = torch.sum(crossentropy_fact.unsqueeze(-1) * grads, dim=(0,1))

        else:
            raise LeaspyAlgoInputError(f"'{model.noise_model}' noise is currently not implemented in 'scipy_minimize' algorithm. "
                                       f"Please open an issue on Gitlab if needed.")

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

    def _get_individual_parameters_patient(self, model, times, values, *, with_jac: bool, patient_id=None):
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
        with_jac : bool
            Should we speed-up the minimization by sending exact gradient of optimized function?
        patient_id : str (or None)
            ID of patient (essentially here for logging purposes when no convergence)

        Returns
        -------
        individual parameters : dict[str, :class:`torch.Tensor` [1,n_dims_param]]
            Individual parameters as a dict of tensors.
        reconstruction error : :class:`torch.Tensor` [n_tpts, n_features]
            Model values minus real values.
        """

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
            # log full results if optimization failed
            res['reconstruction_mae'] = err_f.abs().mean().item() # all tpts & fts instead of mean?
            res['individual_parameters'] = individual_params_f
            self.logger(patient_id, res)

        return individual_params_f, err_f

    def _get_individual_parameters_patient_master(self, it, data, model, *, with_jac: bool, patient_id=None):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        it : int
            The iteration number.
        data : :class:`.Dataset`
            Contains the individual scores.
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        with_jac : bool
            Should we speed-up the minimization by sending exact gradient of optimized function?
        patient_id : str (or None)
            ID of patient (essentially here for logging purposes when no convergence)

        Returns
        -------
        :class:`.IndividualParameters`
            Contains the individual parameters of all patients.
        """
        times = data.get_times_patient(it)  # torch.Tensor[n_tpts]
        values = data.get_values_patient(it)  # torch.Tensor[n_tpts, n_fts]

        individual_params_tensorized, err = self._get_individual_parameters_patient(model, times, values,
                                                                                    with_jac=with_jac, patient_id=patient_id)

        if self.algo_parameters.get('progress_bar', True):
            self._display_progress_bar(it, data.n_individuals, suffix='subjects')

        # transformation is needed because of IndividualParameters expectations...
        return {k: v.item() if k != 'sources' else v.detach().squeeze(0).tolist()
                for k,v in individual_params_tensorized.items()}

    def is_jacobian_implemented(self, model) -> bool:
        """Check that the jacobian of model is implemented."""
        default_individual_params = self._pull_individual_parameters(self._initialize_parameters(model), model)
        empty_tpts = torch.tensor([[]], dtype=torch.float32)
        try:
            model.compute_jacobian_tensorized(empty_tpts, default_individual_params)
            return True
        except NotImplementedError:
            return False

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

        if self.algo_parameters.get('progress_bar', True):
            self._display_progress_bar(-1, data.n_individuals, suffix='subjects')

        # optimize by sending exact gradient of optimized function?
        with_jac = self.algo_parameters['use_jacobian']
        if with_jac and not self.is_jacobian_implemented(model):
            warnings.warn('In `scipy_minimize` you requested `use_jacobian=True` but it is not implemented in your model'
                          f'"{model.name}". Falling back to `use_jacobian=False`...')
            with_jac = False

        ind_p_all = Parallel(n_jobs=self.algo_parameters['n_jobs'])(
            delayed(self._get_individual_parameters_patient_master)(it_pat, data, model, with_jac=with_jac, patient_id=id_pat)
            for it_pat, id_pat in enumerate(data.indices))

        for it_pat, ind_params_pat in enumerate(ind_p_all):
            id_pat = data.indices[it_pat]
            individual_parameters.add_individual_parameters(str(id_pat), ind_params_pat)

        return individual_parameters

