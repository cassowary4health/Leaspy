from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Type
from pprint import pformat
import warnings
from dataclasses import dataclass, field

import torch
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize


from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.variables.specs import VarName, LatentVariable, IndividualLatentVariable
from leaspy.variables.state import State
from leaspy.utils.typing import DictParamsTorch

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


@dataclass(frozen=True)
class _AffineScaling:
    """Affine scaling used for individual latent variables, so that gradients are of the same order of magnitude in scipy minimize."""
    loc: torch.Tensor
    scale: torch.Tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        shape = self.loc.shape
        assert self.scale.shape == shape
        return shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def from_latent_variable(cls, var: LatentVariable, state: State) -> _AffineScaling:
        """Natural scaling for latent variable: (mode, stddev)."""
        return cls(
            var.prior.mode.call(state),
            var.prior.stddev.call(state),
        )


@dataclass
class _AffineScalings1D:
    """Util class to deal with scaled 1D tensors, that are concatenated together in a single 1D tensor (in order)."""
    scalings: Dict[VarName, _AffineScaling]
    slices: Dict[VarName, slice] = field(init=False, repr=False, compare=False)
    length: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        assert all(scl.ndim == 1 for scl in self.scalings.values()), "Individual latent variables should all be 1D vectors"
        dims = {n: scl.shape[0] for n, scl in self.scalings.items()}
        import operator
        from itertools import accumulate
        cumdims = (0,) + tuple(accumulate(dims.values(), operator.add))
        slices = {
            n: slice(cumdims[i], cumdims[i+1])
            for i, n in enumerate(dims)
        }
        object.__setattr__(self, "slices", slices)
        object.__setattr__(self, "length", cumdims[-1])

    def __len__(self) -> int:
        return self.length

    def zeros(self, *, dtype=np.float32, **kws) -> np.ndarray:
        """New concatenated numpy array of scaled values (with good length)."""
        return np.zeros(len(self), dtype=dtype, **kws)

    def pull(self, x: np.ndarray) -> Dict[VarName, torch.Tensor]:
        """Pull dictionary of values (in their natural scale) from the concatenated 1D tensor of scaled values provided."""
        return {
            # unsqueeze 1 dimension at left
            n: scl.loc + scl.scale * torch.tensor(x[None, self.slices[n]]).float()
            for n, scl in self.scalings.items()
        }

    @classmethod
    def from_state(cls, state: State, var_type: Type[LatentVariable]) -> _AffineScalings1D:
        """Get the affine scalings of latent variables so their gradients have the same order of magnitude during optimization."""
        return cls({
            var_name: _AffineScaling.from_latent_variable(var, state)
            for var_name, var in state.dag.sorted_variables_by_type[var_type].items()
        })


class ScipyMinimize(AbstractPersonalizeAlgo):
    """
    Gradient descent based algorithm to compute individual parameters,
    `i.e.` personalize a model to a given set of subjects.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
        In particular the parameter `custom_scipy_minimize_params` may contain
        keyword arguments passed to :func:`scipy.optimize.minimize`.

    Attributes
    ----------
    scipy_minimize_params : dict
        Keyword arguments to be passed to :func:`scipy.optimize.minimize`.
        A default setting depending on whether using jacobian or not is applied
        (cf. `ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN`
         and `ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN`).
        You may customize it by setting the `custom_scipy_minimize_params` algorithm parameter.

    format_convergence_issues : str
        Formatting of convergence issues.
        It should be a formattable string using any of those variables:
           * patient_id: str
           * optimization_result_pformat: str
           * (optimization_result_obj: dict-like)
        cf. `ScipyMinimize.DEFAULT_FORMAT_CONVERGENCE_ISSUES` for the default format.
        You may customize it by setting the `custom_format_convergence_issues` algorithm parameter.

    logger : None or callable str -> None
        The function used to display convergence issues returned by :func:`scipy.optimize.minimize`.
        By default we print the convergences issues if and only if we do not use BFGS optimization method.
        You can customize it at initialization by defining a `logger` attribute to your `AlgorithmSettings` instance.
    """

    name = 'scipy_minimize'

    DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN = {
        'method': "BFGS",
        'options': {
            'gtol': 1e-2,
            'maxiter': 200,
        },
    }
    DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN = {
        'method': "Powell",
        'options': {
            'xtol': 1e-4,
            'ftol': 1e-4,
            'maxiter': 200,
        },
    }
    DEFAULT_FORMAT_CONVERGENCE_ISSUES = "<!> {patient_id}:\n{optimization_result_pformat}"

    regularity_factor: float = 1.

    def __init__(self, settings):

        super().__init__(settings)

        self.scipy_minimize_params = self.algo_parameters.get("custom_scipy_minimize_params", None)
        if self.scipy_minimize_params is None:
            if self.algo_parameters['use_jacobian']:
                self.scipy_minimize_params = self.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN
            else:
                self.scipy_minimize_params = self.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN

        self.format_convergence_issues = self.algo_parameters.get("custom_format_convergence_issues", None)
        if self.format_convergence_issues is None:
            self.format_convergence_issues = self.DEFAULT_FORMAT_CONVERGENCE_ISSUES

        # use a sentinel object to be able to set a custom logger=None
        _sentinel = object()
        self.logger = getattr(settings, 'logger', _sentinel)
        if self.logger is _sentinel:
            self.logger = self._default_logger

    def _default_logger(self, msg: str) -> None:
        # we dynamically retrieve the method of `scipy_minimize_params` so that if we requested jacobian
        # but had to fall back to without jacobian we do print messages!
        if not self.scipy_minimize_params.get('method', 'BFGS').upper() == 'BFGS':
            print('\n' + msg + '\n')

    def _get_normalized_grad_tensor_from_grad_dict(self, dict_grad_tensors: DictParamsTorch, model: AbstractModel):
        """
        From a dict of gradient tensors per param (without normalization),
        returns the full tensor of gradients (= for all params, consecutively):
            * concatenated with conventional order of x0
            * normalized because we derive w.r.t. "standardized" parameter (adimensional gradient)
        """
        raise NotImplementedError("TODO...")
        to_cat = [
            dict_grad_tensors['xi'] * model.parameters['xi_std'],
            dict_grad_tensors['tau'] * model.parameters['tau_std']
        ]
        if 'univariate' not in model.name and model.source_dimension > 0:
            to_cat.append( dict_grad_tensors['sources'] * model.parameters['sources_std'] )

        return torch.cat(to_cat, dim=-1) # 1 individual at a time

    def _get_regularity(self, model: AbstractModel, individual_parameters: DictParamsTorch):
        """
        Compute the regularity term (and its gradient) of a patient given his individual parameters for a given model.

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
        d_regularity, d_regularity_grads = model.compute_regularity_individual_parameters(individual_parameters)
        tot_regularity = sum(d_regularity.values(), 0.)

        return tot_regularity, d_regularity_grads

    def obj_no_jac(self, x: np.ndarray, state: State, scaling: _AffineScalings1D) -> float:
        """
        Objective loss function to minimize in order to get patient's individual parameters

        Parameters
        ----------
        x : numpy.ndarray
            Individual **standardized** parameters
            At initialization x is full of zeros (mode of priors, scaled by std-dev)
        state : :class:`.State`
            The cloned model state that is dedicated to the current individual.
            In particular, individual data variables for the current individual are already loaded into it.
        scaling : _AffineScalings1D
            The scaling to be used for individual latent variables.

        Returns
        -------
        objective : float
            Value of the loss function (negative log-likelihood).
        """
        ips = scaling.pull(x)
        for ip, ip_val in ips.items():
            state[ip] = ip_val

        loss = state["nll_attach"] + self.regularity_factor * state["nll_regul_ind_sum"]
        return loss.item()

    def obj_with_jac(self, x: np.ndarray, state: State, scaling: _AffineScalings1D) -> Tuple[float, torch.Tensor]:
        """
        Objective loss function to minimize in order to get patient's individual parameters, together with its jacobian w.r.t to each of `x` dimension.

        Parameters
        ----------
        x : numpy.ndarray
            Individual **standardized** parameters
            At initialization x is full of zeros (mode of priors, scaled by std-dev)
        state : :class:`.State`
            The cloned model state that is dedicated to the current individual.
            In particular, individual data variables for the current individual are already loaded into it.
        scaling : _AffineScalings1D
            The scaling to be used for individual latent variables.

        Returns
        -------
        2-tuple (as expected by :func:`scipy.optimize.minimize` when ``jac=True``)
            * objective : float
            * gradient : array-like[float] with same length as `x` (= all dimensions of individual latent variables, concatenated)
        """
        raise NotImplementedError("TODO...")

        individual_parameters = self._pull_individual_parameters(x, model)
        predictions = model.compute_individual_tensorized(dataset.timepoints, individual_parameters)

        nll_regul, d_nll_regul_grads = self._get_regularity(model, individual_parameters)
        nll_attach = model.noise_model.compute_nll(dataset, predictions, with_gradient=with_gradient)
        if with_gradient:
            nll_attach, nll_attach_grads_fact = nll_attach

        # we must sum separately the terms due to implicit broadcasting
        nll = nll_attach.squeeze(0).sum() + nll_regul.squeeze(0)

        if not with_gradient:
            return nll.item()

        nll_regul_grads = self._get_normalized_grad_tensor_from_grad_dict(d_nll_regul_grads, model).squeeze(0)

        d_preds_grads = model.compute_jacobian_tensorized(dataset.timepoints, individual_parameters)
        # put derivatives consecutively in the right order
        # --> output shape [1, n_tpts, n_fts [, n_ordinal_lvls], n_dims_params]
        preds_grads = self._get_normalized_grad_tensor_from_grad_dict(d_preds_grads, model).squeeze(0)

        grad_dims_to_sum = tuple(range(0, preds_grads.ndim - 1))
        nll_attach_grads = (preds_grads * nll_attach_grads_fact.squeeze(0).unsqueeze(-1)).sum(dim=grad_dims_to_sum)

        nll_grads = nll_attach_grads + nll_regul_grads

        return nll.item(), nll_grads

    def _get_individual_parameters_patient(self, state: State, *, scaling: _AffineScalings1D, with_jac: bool, patient_id: str):
        """
        Compute the individual parameter by minimizing the objective loss function with scipy solver.

        Parameters
        ----------
        state : :class:`.State`
            The cloned model state that is dedicated to the current individual.
            In particular, data variables for the current individual are already loaded into it.
        scaling : _AffineScalings1D
            The scaling to be used for individual latent variables.
        with_jac : bool
            Should we speed-up the minimization by sending exact gradient of optimized function?
        patient_id : str
            ID of patient (essentially here for logging purposes when no convergence)

        Returns
        -------
        pyt_individual_params : dict[str, :class:`torch.Tensor` [1,n_dims_param]]
            Individual parameters as a dict of tensors.
        reconstruction_loss : :class:`torch.Tensor`
            Model canonical loss (content & shape depend on noise model).
            TODO
        """

        obj = self.obj_with_jac if with_jac else self.obj_no_jac
        res = minimize(
            obj,
            jac=with_jac,
            x0=scaling.zeros(),
            args=(state, scaling),
            **self.scipy_minimize_params
        )

        pyt_individual_params = scaling.pull(res.x)
        # TODO/WIP: we may want to return residuals MAE or RMSE instead (since nll is not very interpretable...)
        #loss = model.compute_canonical_loss_tensorized(patient_dataset, pyt_individual_params)
        loss = self.obj_no_jac(res.x, state, scaling)

        if not res.success and self.logger:
            # log full results if optimization failed
            # including mean of reconstruction loss for this subject on all his personalization visits, but per feature
            res['reconstruction_loss'] = loss
            res['individual_parameters'] = pyt_individual_params

            cvg_issue = self.format_convergence_issues.format(
                patient_id=patient_id,
                optimization_result_obj=res,
                optimization_result_pformat=pformat(res, indent=1),
            )
            self.logger(cvg_issue)

        return pyt_individual_params, loss

    def _get_individual_parameters_patient_master(
        self,
        state: State,
        *,
        scaling: _AffineScalings1D,
        progress: Tuple[int, int],
        with_jac: bool,
        patient_id: str,
    ):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        state : :class:`.State`
            The cloned model state that is dedicated to the current individual.
            In particular, individual data variables for the current individual are already loaded into it.
        scaling : _AffineScalings1D
            The scaling to be used for individual latent variables.
        progress : tuple[int >= 0, int > 0]
            Current progress in loop (n, out-of-N).
        with_jac : bool
            Should we speed-up the minimization by sending exact gradient of optimized function?
        patient_id : str
            ID of patient (essentially here for logging purposes when no convergence)

        Returns
        -------
        :class:`.IndividualParameters`
            Contains the individual parameters of all patients.
        """
        individual_params_tensorized, _ = self._get_individual_parameters_patient(
            state, scaling=scaling, with_jac=with_jac, patient_id=patient_id
        )

        if self.algo_parameters.get('progress_bar', True):
            self._display_progress_bar(*progress, suffix='subjects')

        # TODO/WIP: change this really dirty stuff (hardcoded...)
        # transformation is needed because of current `IndividualParameters` expectations... --> change them
        return {
            k: v.item() if k != 'sources' else v.detach().squeeze(0).tolist()
            for k, v in individual_params_tensorized.items()
        }

    def is_jacobian_implemented(self, model: AbstractModel) -> bool:
        """Check that the jacobian of model is implemented."""
        # TODO/WIP: quick hack for now
        return any("jacobian" in var_name for var_name in model.dag)
        #default_individual_params = self._pull_individual_parameters(self._initialize_parameters(model), model)
        #empty_tpts = torch.tensor([[]], dtype=torch.float32)
        #try:
        #    model.compute_jacobian_tensorized(empty_tpts, default_individual_params)
        #    return True
        #except NotImplementedError:
        #    return False

    def _get_individual_parameters(self, model: AbstractModel, dataset: Dataset):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        dataset : :class:`.Dataset` class object
            Contains the individual scores.

        Returns
        -------
        :class:`.IndividualParameters`
            Contains the individual parameters of all patients.
        """

        # Easier to pass a Dataset with 1 individual rather than individual times, values
        # to avoid duplicating code in noise model especially
        df = dataset.to_pandas()
        import pandas as pd
        assert pd.api.types.is_string_dtype(df.index.dtypes["ID"]), "Individuals ID should be strings"

        data = Data.from_dataframe(df, drop_full_nan=False, warn_empty_column=False)
        datasets = {
            idx: Dataset(data[[idx]], no_warning=True)
            for idx in dataset.indices
        }

        # Fetch model internal state (latent pop. vars should be OK)
        state = model._state
        assert state is not None, "State was not properly initialized"
        # Fixed scalings for individual parameters
        ips_scalings = _AffineScalings1D.from_state(state, var_type=IndividualLatentVariable)

        # Clone model states (1 per individual with the appropriate dataset loaded into each of them)
        states = {}
        for idx in dataset.indices:
            state_pat = state.clone()
            state_pat.auto_fork_type = None
            model.put_data_variables(state_pat, datasets[idx])
            states[idx] = state_pat

        if self.algo_parameters.get('progress_bar', True):
            self._display_progress_bar(-1, dataset.n_individuals, suffix='subjects')

        # optimize by sending exact gradient of optimized function?
        with_jac = self.algo_parameters['use_jacobian']
        if with_jac and not self.is_jacobian_implemented(model):
            warnings.warn('In `scipy_minimize` you requested `use_jacobian=True` but it is not implemented in your model'
                          f'"{model.name}". Falling back to `use_jacobian=False`...')
            with_jac = False
            if self.algo_parameters.get("custom_scipy_minimize_params", None) is None:
                # reset default `scipy_minimize_params`
                self.scipy_minimize_params = self.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN
            # TODO? change default logger as well?

        ind_p_all = Parallel(n_jobs=self.algo_parameters['n_jobs'])(
            delayed(self._get_individual_parameters_patient_master)(
                state_pat,
                scaling=ips_scalings,
                progress=(it_pat, dataset.n_individuals),
                with_jac=with_jac,
                patient_id=id_pat,
            )
            # TODO use Parallel + tqdm instead of custom progress bar...
            for it_pat, (id_pat, state_pat) in enumerate(states.items())
        )

        individual_parameters = IndividualParameters()
        for id_pat, ind_params_pat in zip(dataset.indices, ind_p_all):
            individual_parameters.add_individual_parameters(str(id_pat), ind_params_pat)

        return individual_parameters
