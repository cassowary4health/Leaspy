import json
import torch


from leaspy import __version__

from abc import ABC, abstractmethod
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.models.utils.noise_model import NoiseModel
from leaspy.io.realizations.collection_realization import CollectionRealization
from leaspy.io.realizations.realization import Realization

from leaspy.utils.typing import DictParamsTorch

from leaspy.utils.typing import Optional
from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

from leaspy.utils.typing import FeatureType, KwargsType, DictParams, DictParamsTorch, Union, List, Dict, Tuple, Iterable, Optional



# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class SurvivalModel(ABC):
    """
    Univariate (logistic or linear) model for a single variable of interest.

    Parameters
    ----------
    name : str
        Name of the model
    **kwargs
        Hyperparameters of the model

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    SUBTYPES_SUFFIXES = {
        'univariate_survival_weibull': '_weibull',
    }

    def __init__(self, name: str, **kwargs):

        self.is_initialized: bool = False
        self.name = name
        self.features: List[FeatureType] = None
        self.univariate = True
        self.dimension = 1
        self.source_dimension = 0  # TODO, None ???
        self.noise_model = 'survival'

        self.parameters = {
            "rho": None,
            "nu": None,
            "xi_mean": None,
            "xi_std": None,
             "noise_std": None

        }
        self.bayesian_priors = None
        self.attributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                # for logistic: "p0" = 1 / (1+exp(g)) i.e. exp(g) = 1/p0 - 1
                # for linear: "p0" = g
                'rho_std': None,
                'nu_std': None,
            }
        }

        # subtype of univariate model
        self._subtype_suffix = self._check_subtype()

        # Load hyperparameters at end to overwrite default for new hyperparameters
        self.load_hyperparameters(kwargs)

    @classmethod
    def _raise_if_unknown_hyperparameters(cls, known_hps: Iterable[str], given_hps: KwargsType) -> None:
        """Helper function raising a :exc:`.LeaspyModelInputError` if any unknown hyperparameter provided for model."""
        # TODO: replace with better logic from GenericModel in the future
        unexpected_hyperparameters = set(given_hps.keys()).difference(known_hps)
        if len(unexpected_hyperparameters) > 0:
            raise LeaspyModelInputError(
                f"Only {known_hps} are valid hyperparameters for {cls.__qualname__}. "
                f"Unknown hyperparameters provided: {unexpected_hyperparameters}.")

    def _audit_individual_parameters(self, ips: DictParams) -> KwargsType:
        """
        Perform various consistency and compatibility (with current model) checks
        on an individual parameters dict and outputs qualified information about it.

        TODO? move to IndividualParameters class?

        Parameters
        ----------
        ips : dict[param: str, Any]
            Contains some un-trusted individual parameters.
            If representing only one individual (in a multivariate model) it could be:
                * {'tau':0.1, 'xi':-0.3, 'sources':[0.1,...]}

            Or for multiple individuals:
                * {'tau':[0.1,0.2,...], 'xi':[-0.3,0.2,...], 'sources':[[0.1,...],[0,...],...]}

            In particular, a sources vector (if present) should always be a array_like, even if it is 1D

        Returns
        -------
        ips_info : dict
            * ``'nb_inds'`` : int >= 0
                number of individuals present
            * ``'tensorized_ips'`` : dict[param:str, `torch.Tensor`]
                tensorized version of individual parameters
            * ``'tensorized_ips_gen'`` : generator
                generator providing tensorized individual parameters for all individuals present (ordered as is)

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            if any of the consistency/compatibility checks fail
        """

        def is_array_like(v):
            # abc.Collection is useless here because set, np.array(scalar) or torch.tensor(scalar)
            # are abc.Collection but are not array_like in numpy/torch sense or have no len()
            try:
                len(v) # exclude np.array(scalar) or torch.tensor(scalar)
                return hasattr(v, '__getitem__') # exclude set
            except Exception:
                return False

        # Model supports and needs sources?
        has_sources = hasattr(self, 'source_dimension') and isinstance(self.source_dimension, int) and self.source_dimension > 0

        # Check parameters names
        expected_parameters = set(['xi', 'tau'] + int(has_sources)*['sources'])
        given_parameters = set(ips.keys())
        symmetric_diff = expected_parameters.symmetric_difference(given_parameters)
        if len(symmetric_diff) > 0:
            raise LeaspyIndividualParamsInputError(
                    f'Individual parameters dict provided {given_parameters} '
                    f'is not compatible for {self.name} model. '
                    f'The expected individual parameters are {expected_parameters}.')

        # Check number of individuals present (with low constraints on shapes)
        ips_is_array_like = {k: is_array_like(v) for k,v in ips.items()}
        ips_size = {k: len(v) if ips_is_array_like[k] else 1 for k,v in ips.items()}

        if has_sources:
            s = ips['sources']

            if not ips_is_array_like['sources']:
                raise LeaspyIndividualParamsInputError(f'Sources must be an array_like but {s} was provided.')

            tau_xi_scalars = all(ips_size[k] == 1 for k in ['tau','xi'])
            if tau_xi_scalars and (ips_size['sources'] > 1):
                # is 'sources' not a nested array? (allowed iff tau & xi are scalars)
                if not is_array_like(s[0]):
                    # then update sources size (1D vector representing only 1 individual)
                    ips_size['sources'] = 1

            # TODO? check source dimension compatibility?

        uniq_sizes = set(ips_size.values())
        if len(uniq_sizes) != 1:
            raise LeaspyIndividualParamsInputError('Individual parameters sizes are not compatible together. '
                                                  f'Sizes are {ips_size}.')

        # number of individuals present
        n_inds = uniq_sizes.pop()

        # properly choose unsqueezing dimension when tensorizing array_like (useful for sources)
        unsqueeze_dim = -1 # [1,2] => [[1],[2]] (expected for 2 individuals / 1D sources)
        if n_inds == 1:
            unsqueeze_dim = 0 # [1,2] => [[1,2]] (expected for 1 individual / 2D sources)

        # tensorized (2D) version of ips
        t_ips = {k: self._tensorize_2D(v, unsqueeze_dim=unsqueeze_dim) for k,v in ips.items()}

        # construct logs
        return {
            'nb_inds': n_inds,
            'tensorized_ips': t_ips,
            'tensorized_ips_gen': ({k: v[i,:].unsqueeze(0) for k,v in t_ips.items()} for i in range(n_inds))
        }

    def _check_subtype(self):
        if self.name not in self.SUBTYPES_SUFFIXES.keys():
            raise LeaspyModelInputError(f'Survival model name should be among these valid sub-types: '
                                        f'{list(self.SUBTYPES_SUFFIXES.keys())}.')

        return self.SUBTYPES_SUFFIXES[self.name]

    def save(self, path: str, **kwargs):

        model_parameters_save = self.parameters.copy()
        for key, value in model_parameters_save.items():
            if isinstance(value, torch.Tensor):
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            #'dimension': 1,
            'noise_model': self.noise_model,
            'parameters': model_parameters_save
        }

        # TODO : in leaspy models there should be a method to only return the dict describing the model
        # and then another generic method (inherited) should save this dict
        # (with extra standard fields such as 'leaspy_version' for instance)

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_hyperparameters(self, hyperparameters: dict):

        expected_hyperparameters = ('features',)
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']

        # load new `noise_model` directly in-place & add the recognized hyperparameters to known tuple
        # TODO? forbid the usage of `gaussian_diagonal` noise for such model?
        expected_hyperparameters += NoiseModel.set_noise_model_from_hyperparameters(self, hyperparameters)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def initialize(self, dataset, method="default"):

        self.features = dataset.headers

        self.parameters = initialize_parameters(self, dataset, method)
        self.attributes = AttributesFactory.attributes(self.name, dimension=1)

        # Postpone the computation of attributes when really needed!
        #self.attributes.update(['all'], self.parameters)

        self.is_initialized = True

    def load_parameters(self, parameters):
        self.parameters = {}

        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k])

        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)

    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model
        """
        # TODO to move in the MCMC-fit algorithm
        self.MCMC_toolbox = {
            'priors': {'rho_std': 0.01,
                       'nu_std': 0.01}, # population parameter
        }


        self.MCMC_toolbox['attributes'] = AttributesFactory.attributes(self.name, dimension=1)
        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    ##########
    # CORE
    ##########
    def update_MCMC_toolbox(self, vars_to_update, realizations):
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        TODO to move in the MCMC-fit algorithm

        Parameters
        ----------
        vars_to_update : container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        print("MCMC",vars_to_update, realizations['rho'].tensor_realizations)
        values = {}
        if any(c in vars_to_update for c in ('rho', 'all')):
            values['rho'] = realizations['rho'].tensor_realizations

        if any(c in vars_to_update for c in ('nu', 'nu_collinear', 'all')):
            values['nu'] = realizations['nu'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(vars_to_update, values)

    def _call_method_from_attributes(self, method_name: str, attribute_type: Optional[str], **call_kws):
        # TODO: move in a abstract parent class for univariate & multivariate models (like AbstractManifoldModel...)
        if attribute_type is None:
            return getattr(self.attributes, method_name)(**call_kws)
        elif attribute_type == 'MCMC':
            return getattr(self.MCMC_toolbox['attributes'], method_name)(**call_kws)
        else:
            raise LeaspyModelInputError(f"The specified attribute type does not exist: {attribute_type}. "
                                        "Should be None or 'MCMC'.")

    def _get_attributes(self, attribute_type: Optional[str]):
        return self._call_method_from_attributes('get_attributes', attribute_type)

    def compute_mean_traj(self, timepoints, *, attribute_type: Optional[str] = None):
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?
        TODO generalize in abstract manifold model

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` [1, n_timepoints]
        attribute_type : 'MCMC' or None

        Returns
        -------
        :class:`torch.Tensor` [1, n_timepoints, dimension]
            The group-average values at given timepoints
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']]),
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters, attribute_type=attribute_type)

    ##################
    #

    def get_population_realization_names(self) -> List[str]:
        """
        Get names of population variables of the model.

        Returns
        -------
        list[str]
        """
        return [name for name, value in self.random_variable_informations().items()
                if value['type'] == 'population']

    def get_individual_realization_names(self) -> List[str]:
        """
        Get names of individual variables of the model.

        Returns
        -------
        list[str]
        """
        return [name for name, value in self.random_variable_informations().items()
                if value['type'] == 'individual']

    def __str__(self):
        output = "=== MODEL ==="
        for p, v in self.parameters.items():
            if isinstance(v, float) or (hasattr(v, 'ndim') and v.ndim == 0):
                # for 0D tensors / arrays the default behavior is to print all digits...
                # change this!
                v_repr = f'{v:.{1+torch_print_opts.precision}g}'
            else:
                # torch.tensor, np.array, ...
                # in particular you may use `torch.set_printoptions` and `np.set_printoptions` globally
                # to tune the number of decimals when printing tensors / arrays
                v_repr = str(v)
                # remove tensor prefix & possible dtype suffix
                v_repr = re.sub(r'^[^\(]+\(', '', v_repr)
                v_repr = re.sub(r'(?:, dtype=.+)?\)$', '', v_repr)
                # adjust justification
                spaces = " "*len(f"{p} : [")
                v_repr = re.sub(r'\n[ ]+\[', f'\n{spaces}[', v_repr)

            output += f"\n{p} : {v_repr}"
        return output

    def compute_regularity_realization(self, realization: Realization):
        """
        Compute regularity term for a :class:`.Realization` instance.

        Parameters
        ----------
        realization : :class:`.Realization`

        Returns
        -------
        :class:`torch.Tensor` of the same shape as `realization.tensor_realizations`
        """
        if realization.variable_type == 'population':
            # Regularization of population variables around current model values
            mean = self.parameters[realization.name]
            std = self.MCMC_toolbox['priors'][f"{realization.name}_std"]
        elif realization.variable_type == 'individual':
            # Regularization of individual parameters around mean / std from model parameters
            mean = self.parameters[f"{realization.name}_mean"]
            std = self.parameters[f"{realization.name}_std"]
        else:
            raise LeaspyModelInputError(f"Variable type '{realization.variable_type}' not known, should be 'population' or 'individual'.")

        # we do not need to include regularity constant (priors are always fixed at a given iteration)
        return self.compute_regularity_variable(realization.tensor_realizations, mean, std, include_constant=False)

    def compute_regularity_variable(self, value: torch.FloatTensor, mean: torch.FloatTensor, std: torch.FloatTensor,
                                    *, include_constant: bool = True) -> torch.FloatTensor:
        """
        Compute regularity term (Gaussian distribution), low-level.

        TODO: should be encapsulated in a RandomVariableSpecification class together with other specs of RV.

        Parameters
        ----------
        value, mean, std : :class:`torch.Tensor` of same shapes
        include_constant : bool (default True)
            Whether we include or not additional terms constant with respect to `value`.

        Returns
        -------
        :class:`torch.Tensor` of same shape than input
        """
        # This is really slow when repeated on tiny tensors (~3x slower than direct formula!)
        #return -self.regularization_distribution_factory(mean, std).log_prob(value)

        y = (value - mean) / std
        neg_loglike = 0.5*y*y
        if include_constant:
            neg_loglike += 0.5*torch.log(TWO_PI * std**2)

        return neg_loglike

    def initialize_realizations_for_model(self, n_individuals: int, **init_kws) -> CollectionRealization:
        """
        Initialize a :class:`.CollectionRealization` used during model fitting or mode/mean realization personalization.

        Parameters
        ----------
        n_individuals : int
            Number of individuals to track
        **init_kws
            Keyword arguments passed to :meth:`.CollectionRealization.initialize`.
            (In particular `individual_variable_init_at_mean` to "initialize at mean" or `skip_variable` to filter some variables)

        Returns
        -------
        :class:`.CollectionRealization`
        """
        realizations = CollectionRealization()
        realizations.initialize(n_individuals, self, **init_kws)
        return realizations

    def smart_initialization_realizations(self, dataset, realizations: CollectionRealization) -> CollectionRealization:
        """
        Smart initialization of realizations if needed (input may be modified in-place).

        Default behavior to return `realizations` as they are (no smart trick).

        Parameters
        ----------
        dataset : :class:`.Dataset`
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        :class:`.CollectionRealization`
        """
        return realizations

    def _create_dictionary_of_population_realizations(self):
        pop_dictionary: Dict[str, Realization] = {}
        for name_var, info_var in self.random_variable_informations().items():
            if info_var['type'] != "population":
                continue
            real = Realization.from_tensor(name_var, info_var['shape'], info_var['type'], self.parameters[name_var])
            pop_dictionary[name_var] = real

        return pop_dictionary


    def get_param_from_real(self, realizations: CollectionRealization) -> DictParamsTorch:
        """
        Get individual parameters realizations from all model realizations

        <!> The tensors are not cloned and so a link continue to exist between the individual parameters
            and the underlying tensors of realizations.

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`

        Returns
        -------
        dict[param_name: str, :class:`torch.Tensor` [n_individuals, dims_param]]
            Individual parameters
        """
        return {
            variable_ind: realizations[variable_ind].tensor_realizations
            for variable_ind in self.get_individual_realization_names()
        }

    @staticmethod
    def _compute_std_from_var(variance: torch.FloatTensor, *, varname: str, tol: float = 1e-5) -> torch.FloatTensor:
        """
        Check that variance is strictly positive and return its square root, otherwise fail with a convergence error.

        If variance is multivariate check that all components are strictly positive.

        TODO? a full Bayesian setting with good priors on all variables should prevent such convergence issues.

        Parameters
        ----------
        var : :class:`torch.Tensor`
            The variance we would like to convert to a std-dev.
        varname : str
            The name of the variable - to display a nice error message.
        tol : float
            The lower bound on variance, under which the converge error is raised.

        Returns
        -------
        torch.FloatTensor

        Raises
        ------
        :exc:`.LeaspyConvergenceError`
        """
        if (variance < tol).any():
            raise LeaspyConvergenceError(f"The parameter '{varname}' collapsed to zero, which indicates a convergence issue.\n"
                                         "Start by investigating what happened in the logs of your calibration and try to double check:"
                                         "\n- your training dataset (not enough subjects and/or visits? too much missing data?)"
                                         "\n- the hyperparameters of your Leaspy model (`source_dimension` too low or too high? "
                                         "`noise_model` not suited to your data?)"
                                         "\n- the hyperparameters of your calibration algorithm"
                                        )

        return variance.sqrt()

    def move_to_device(self, device: torch.device) -> None:
        """
        Move a model and its relevant attributes to the specified device.

        Parameters
        ----------
        device : torch.device
        """

        # Note that in a model, the only tensors that need offloading to a
        # particular device are in the model.parameters dict as well as in the
        # attributes and MCMC_toolbox['attributes'] objects

        for parameter in self.parameters:
            self.parameters[parameter] = self.parameters[parameter].to(device)

        if hasattr(self, "attributes"):
            self.attributes.move_to_device(device)

        if hasattr(self, "MCMC_toolbox"):
            MCMC_toolbox_attributes = self.MCMC_toolbox.get("attributes", None)
            if MCMC_toolbox_attributes is not None:
                MCMC_toolbox_attributes.move_to_device(device)


    #################@
    # Value compute

    def compute_individual_tensorized(self, data, individual_parameters, *, attribute_type=None):

            # Get parameters
            rho, nu = self._get_attributes(attribute_type)

            # Get Individual parameters
            xi = individual_parameters['xi'].reshape(data.event_time_min.shape)

            # Reparametrized survival
            reparametrized_time_min = (torch.exp(xi) * data.event_time_min)
            reparametrized_time_max = (torch.exp(xi) * data.event_time_max)

            # Survival
            survival = torch.exp(-(reparametrized_time_min * nu) ** rho)
            print(rho, nu)

            # Hazard only for patient with event not censored
            hazard = (rho * nu) * ((reparametrized_time_max * nu) ** (rho - 1))
            hazard = torch.nan_to_num((data.mask_event * hazard), nan=1.)

            return hazard * survival


    ######################
    # TO CHANGE
    """
    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):

        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # get tensorized attributes
        g = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        # compute age
        ages = torch.exp(-xi) * torch.log(g/(1 / value - 1)) + tau
        assert ages.shape == value.shape

        return ages

    
    @suffixed_method
    def compute_jacobian_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_jacobian_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)
        alpha = torch.exp(xi).unsqueeze(-1)

        # Jacobian of model expected value w.r.t. individual parameters
        derivatives = {
            'xi': reparametrized_time.unsqueeze(-1),
            'tau': (-alpha * torch.ones_like(reparametrized_time)).unsqueeze(-1),
        }

        return derivatives

    def compute_jacobian_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        # Log likelihood computation
        LL = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features==1)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            LL = LL.unsqueeze(-1)
            g = g.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas.cumsum(dim=-1)

        model = 1. / (1. + g * torch.exp(-LL))

        c = model * (1. - model)

        derivatives = {
            'xi': (reparametrized_time).unsqueeze(-1).unsqueeze(-1),
            'tau': (-alpha).unsqueeze(-1),
        }

        if self.is_ordinal:
            ordinal_lvls_shape = c.shape[-1]
            for param in derivatives:
                derivatives[param] = derivatives[param].unsqueeze(-2).repeat(1, 1, 1, ordinal_lvls_shape, 1)

        for param in derivatives:
            derivatives[param] = c.unsqueeze(-1) * derivatives[param]

        # Compute derivative of the pdf and not of the sf
        if self.noise_model == 'ordinal':
            for param in derivatives:
                derivatives[param] = self.compute_ordinal_pdf_from_ordinal_sf(derivatives[param])

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts == 1 [, extra_dim_ordinal_models], n_dims_param)]
        return derivatives
    """

    ##################
    # STATISTICS

    def compute_individual_attachment_tensorized(self, data, param_ind: DictParamsTorch, *,
                                                 attribute_type) -> torch.FloatTensor:
        """
        Compute attachment term (per subject)

        Parameters
        ----------
        data : :class:`.Dataset`
            Contains the data of the subjects, in particular the subjects' time-points and the mask for nan values & padded visits

        param_ind : dict
            Contain the individual parameters

        attribute_type : Any
            Flag to ask for MCMC attributes instead of model's attributes.

        Returns
        -------
        attachment : :class:`torch.Tensor`
            Negative Log-likelihood, shape = (n_subjects,)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If invalid `noise_model` for model
        """

        # TODO: this snippet could be implemented directly in NoiseModel (or subclasses depending on noise structure)
        if self.noise_model is None:
            raise LeaspyModelInputError('`noise_model` was not set correctly set.')

        elif 'survival' in self.noise_model:
            attachment = self.compute_individual_tensorized(data, param_ind, attribute_type=attribute_type)

        else:
            raise LeaspyModelInputError(f'`noise_model` should be in {NoiseModel.VALID_NOISE_STRUCTS}')

        return -torch.log(attachment)

    def _center_xi_realizations(self, realizations):
        # This operation does not change the orthonormal basis
        # (since the resulting v0 is collinear to the previous one)
        # Nor all model computations (only v0 * exp(xi_i) matters),
        # it is only intended for model identifiability / `xi_i` regularization
        # <!> all operations are performed in "log" space (v0 is log'ed)
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['nu'].tensor_realizations = realizations['nu'].tensor_realizations + mean_xi

        #self.update_MCMC_toolbox(['nu_collinear'], realizations)

        return realizations

    def compute_sufficient_statistics(self, data, realizations):

        # modify realizations in-place
        realizations = self._center_xi_realizations(realizations)

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone_realizations()
        sufficient_statistics = {}
        sufficient_statistics['rho'] = realizations['rho'].tensor_realizations
        sufficient_statistics['nu'] = realizations['nu'].tensor_realizations
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)
        print("suff", sufficient_statistics['rho'])
        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        individual_parameters = self.get_param_from_real(realizations)

        if self.noise_model in ['survival']:
            sufficient_statistics['log-likelihood'] = self.compute_individual_attachment_tensorized(data,
                                                                                                    individual_parameters,
                                                                                                    attribute_type='MCMC').sum()
        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):
        # Memoryless part of the algorithm

        # modify realizations in-place!
        realizations = self._center_xi_realizations(realizations)

        # unlink model parameters from updates in realizations!
        realizations = realizations.clone_realizations()
        print("burn", realizations['rho'].tensor_realizations)
        self.parameters['rho'] = realizations['rho'].tensor_realizations
        self.parameters['nu'] = realizations['nu'].tensor_realizations
        xi = realizations['xi'].tensor_realizations
        #self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)

        param_ind = self.get_param_from_real(realizations)
        if self.noise_model in ['survival']:
            self.parameters['log-likelihood'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                              attribute_type='MCMC').sum()

    def update_model_parameters_normal(self, data, suff_stats):
        # Stochastic sufficient statistics used to update the parameters of the model

        self.parameters['rho'] = suff_stats['rho']
        self.parameters['nu'] = suff_stats['nu']

        xi_mean = self.parameters['xi_mean']
        xi_var_updt = torch.mean(suff_stats['xi_sqrd']) - 2. * xi_mean * torch.mean(suff_stats['xi'])
        xi_var = xi_var_updt + xi_mean ** 2
        self.parameters['xi_std'] = self._compute_std_from_var(xi_var, varname='xi_std')
        #self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        if self.noise_model in ['survival']:
            self.parameters['log-likelihood'] = suff_stats['log-likelihood'].sum()

    def random_variable_informations(self):

        ## Population variables
        rho_infos = {
            "name": "rho",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }
        nu_infos = {
            "name": "nu",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        ## Individual variables

        xi_infos = {
            "name": "xi",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "nu": nu_infos,
            "rho": rho_infos,
            "xi": xi_infos,
        }

        return variables_infos
