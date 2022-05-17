import json

import torch

from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.models.utils.noise_model import NoiseModel

from leaspy.utils.typing import Optional
from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class UnivariateModel(AbstractModel):
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
        'univariate_linear': '_linear',
        'univariate_logistic': '_logistic'
    }

    def __init__(self, name: str, **kwargs):

        super().__init__(name)

        self.dimension = 1
        self.source_dimension = 0  # TODO, None ???
        self.noise_model = 'gaussian_scalar'

        self.parameters = {
            "g": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
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
                'g_std': None,
            }
        }

        # subtype of univariate model
        self._subtype_suffix = self._check_subtype()

        # Load hyperparameters at end to overwrite default for new hyperparameters
        self.load_hyperparameters(kwargs)

    def _check_subtype(self):
        if self.name not in self.SUBTYPES_SUFFIXES.keys():
            raise LeaspyModelInputError(f'Univariate model name should be among these valid sub-types: '
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

        if self.noise_model == 'ordinal':
            model_settings['batch_deltas_ordinal'] = self.ordinal_infos["batch_deltas"]

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

        if self.noise_model == 'ordinal':
            if "linear" in self.name:
                raise LeaspyModelInputError("Noise model 'ordinal' is only compatible with 'logistic' and 'univariate_logistic' models")
            if hasattr(self, 'ordinal_infos'):
                self.ordinal_infos["batch_deltas"] = hyperparameters.get('batch_deltas_ordinal',
                                                                         self.ordinal_infos["batch_deltas"])
            else:
                self.ordinal_infos = {"batch_deltas": hyperparameters.get('batch_deltas_ordinal', False),
                                      "max_level": 1,
                                      "features": [],
                                      "mask": 1.,
                                      }
            expected_hyperparameters += ('batch_deltas_ordinal',)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def initialize(self, dataset, method="default"):

        self.features = dataset.headers

        if self.noise_model == 'ordinal':
            ord = self.ordinal_infos
        else:
            ord = None

        self.parameters = initialize_parameters(self, dataset, method)

        self.attributes = AttributesFactory.attributes(self.name, dimension=1, ordinal_infos=ord)

        # Postpone the computation of attributes when really needed!
        #self.attributes.update(['all'], self.parameters)

        self.is_initialized = True

    def load_parameters(self, parameters):
        self.parameters = {}

        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k])
            if 'deltas' in k:
                if not self.ordinal_infos['batch_deltas']:
                    self.ordinal_infos["features"].append({"name": k[7:], "nb_levels": self.parameters[k].shape[0] + 1}) #k[7:] removes the deltas_ to extract the feature's name
                else:
                    # Find ordinal infos from the delta values themselves
                    for i, feat in enumerate(self.features):
                        bool_array = (self.parameters['deltas'][i, :] != 0).int()
                        if 0 in bool_array:
                            lvl = bool_array.argmin().item() + 2
                        else:
                            lvl = bool_array.shape[0] + 1
                        self.ordinal_infos["features"].append({"name": feat, "nb_levels": lvl})

        if any('deltas' in c for c in parameters.keys()):
            self.ordinal_infos["max_level"] = max([feat["nb_levels"] for feat in self.ordinal_infos["features"]])
            # Mask for setting values > max_level per item to zero
            self.ordinal_infos["mask"] = torch.cat([
                torch.cat([
                    torch.ones((1, 1, 1, feat['nb_levels'])),
                    torch.zeros((1, 1, 1, self.ordinal_infos["max_level"] - feat['nb_levels'])),
                ], dim=-1) for feat in self.ordinal_infos["features"]
            ], dim=2)
            ord = self.ordinal_infos
        else:
            ord = None
        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,
                                                       ordinal_infos=ord)
        self.attributes.update(['all'], self.parameters)

    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model
        """
        # TODO to move in the MCMC-fit algorithm
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01}, # population parameter
        }

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                self.MCMC_toolbox['priors']['deltas_std'] = 0.1
            else:
                for feat in self.ordinal_infos["features"]:
                    self.MCMC_toolbox['priors'][f'deltas_{feat["name"]}_std'] = 0.1
            ord = self.ordinal_infos
        else:
            ord = None
        self.MCMC_toolbox['attributes'] = AttributesFactory.attributes(self.name, dimension=1,
                                                                       ordinal_infos=ord)
        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    def _get_deltas(self, attribute_type):
        """

        Parameters
        ----------
        attribute_type: None or 'MCMC'

        Returns
        -------
        The deltas in the ordinal model
        """
        if attribute_type is None:
            return self.attributes.get_deltas()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_deltas()
        else:
            raise ValueError("The specified attribute type does not exist : {}".format(attribute_type))

    ##########
    # CORE
    ##########
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        TODO to move in the MCMC-fit algorithm

        Parameters
        ----------
        name_of_the_variables_that_have_been_changed : container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                if any(c in L for c in ('deltas', 'all')):
                    values['deltas'] = realizations['deltas'].tensor_realizations
            else:
                for feat in self.ordinal_infos["features"]:
                    if any(c in L for c in ('deltas_'+feat["name"], 'all')):
                        values['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

        self.MCMC_toolbox['attributes'].update(L, values)


    def _get_attributes(self, attribute_type: Optional[str]):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise LeaspyModelInputError(f"The specified attribute type does not exist: {attribute_type}. "
                                        "Should be None or 'MCMC'.")

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
            'tau': torch.tensor([self.parameters['tau_mean']]),
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters, attribute_type=attribute_type)

    @suffixed_method
    def compute_individual_tensorized(self, timepoints, individual_parameters, *, attribute_type=None):
        pass

    def compute_likelihood_from_cdf(self, model_values):
        """
        Computes the likelihood of an ordinal model assuming that the model_values are the CDF.

        Parameters
        ----------
        model_values : `torch.Tensor`
            Cumulative distribution values : model_values[..., l] is the proba to be superior or equal to l+1

        Returns
        -------
        likelihood : `torch.Tensor`
            likelihood[..., l] is the proba to be equal to l
        """

        s = list(model_values.shape)
        s[3] = 1
        mask = self.ordinal_infos["mask"]
        if len(s) == 5:  # in the case of gradient we added a dimension
            mask = mask.unsqueeze(-1)
            first_row = torch.zeros(size=tuple(s)).float()  # gradient(P>=0) = 0
        else:
            first_row = torch.ones(size=tuple(s)).float()  # (P>=0) = 1
        model = model_values * mask
        cdf_sup = torch.cat([first_row, model], dim=3)
        last_row = torch.zeros(size=tuple(s)).float()
        cdf_inf = torch.cat([model, last_row], dim=3)
        likelihood = cdf_sup - cdf_inf

        return likelihood

    def compute_individual_tensorized_logistic(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        g = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        LL = reparametrized_time.unsqueeze(-1)

        if self.noise_model == 'ordinal':
            # add an extra dimension for the levels of the ordinal item
            LL = LL.unsqueeze(-1)
            g = g.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)
            ordinal_scale = deltas.shape[-1] + 1
            deltas_ = torch.cat([torch.zeros((deltas.shape[0], 1)), deltas], dim=1)  # (features, max_level)
            deltas_ = deltas_.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas_.cumsum(dim=-1)

# TODO? more efficient & accurate to compute `torch.exp(-LL + log_g)` since we directly sample & stored log_g
        model = 1. / (1. + g * torch.exp(-LL))

        if self.noise_model == 'ordinal':
            model = self.compute_likelihood_from_cdf(model)

        return model

    def compute_individual_tensorized_linear(self, timepoints, individual_parameters, *, attribute_type=None):

        # Population parameters
        positions = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        return positions + reparametrized_time.unsqueeze(-1)

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(self, value: torch.Tensor,
                                                                 individual_parameters: dict, feature: str):
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):
        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # get tensorized attributes
        g = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        # compute age
        ages = torch.exp(-xi) * torch.log(g/(1 / value - 1)) + tau
        assert ages.shape == value.shape

        return ages

    def compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(self, value: torch.Tensor,
                                                                          individual_parameters: dict, feature: str):
        """
        For one individual, compute age(s) breakpoints at which the given features levels are the most likely (given the subject's
        individual parameters).

        Consistency checks are done in the main API layer.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            Contains the biomarker level value(s) of the subject.

        individual_parameters : dict
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like

        feature : str (or None)
            Name of the considered biomarker (optional for univariate models, compulsory for multivariate models).

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is (1, n_values)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if computation is tried on more than 1 individual
        """
        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        # 1/ get attributes
        g = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)

        # 2/ compute age
        ages_0 = tau + (torch.exp(-xi)) * ((g / (g + 1) ** 2) * torch.log(g))
        deltas = self._get_deltas(None)
        delta_max = deltas[feat_ind].sum()
        ages_max = tau + (torch.exp(-xi)) * ((g / (g + 1) ** 2) * torch.log(g) + delta_max)

        timepoints = torch.linspace(ages_0.item(), ages_max.item(), 1000).unsqueeze(0)

        grid = self.compute_individual_tensorized_logistic(timepoints, individual_parameters, attribute_type=None)[:,:,feat_ind,:]

        MLE = grid.argmax(dim=-1)
        index_cross = (MLE.unsqueeze(1) >= value.unsqueeze(-1)).int().argmax(dim=-1)

        return timepoints[0,index_cross]

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

        if self.noise_model == 'ordinal':
            # add an extra dimension for the levels of the ordinal item
            LL = LL.unsqueeze(-1)
            g = g.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)
            ordinal_scale = deltas.shape[-1] + 1
            deltas_ = torch.cat([torch.zeros((deltas.shape[0], 1)), deltas], dim=1)  # (features, max_level)
            deltas_ = deltas_.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas_.cumsum(dim=-1)

        model = 1. / (1. + g * torch.exp(-LL))

        c = model * (1. - model)

        derivatives = {
            'xi': (reparametrized_time).unsqueeze(-1).unsqueeze(-1),
            'tau': (-alpha).unsqueeze(-1),
        }

        if self.noise_model == 'ordinal':
            ordinal_scale = c.shape[-1]
            for param in derivatives:
                derivatives[param] = derivatives[param].unsqueeze(-2).repeat(1, 1, 1, ordinal_scale, 1)

        for param in derivatives:
            derivatives[param] = c.unsqueeze(-1) * derivatives[param]

        # Compute derivative of the likelihood and not of the cdf
        if self.noise_model == 'ordinal':
            for param in derivatives:
                derivatives[param] = self.compute_likelihood_from_cdf(derivatives[param])

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_sufficient_statistics(self, data, realizations):

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone_realizations()

        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations
            else:
                for feat in self.ordinal_infos["features"]:
                    sufficient_statistics['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        individual_parameters = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, individual_parameters, attribute_type='MCMC')

        if self.noise_model in ['gaussian_scalar', 'gaussian_diagonal']:
            data_reconstruction *= data.mask.float() # speed-up computations

            norm_1 = data.values * data_reconstruction #* data.mask.float()
            norm_2 = data_reconstruction * data_reconstruction #* data.mask.float()

            sufficient_statistics['obs_x_reconstruction'] = norm_1 #.sum(dim=2)
            sufficient_statistics['reconstruction_x_reconstruction'] = norm_2 #.sum(dim=2)

        if self.noise_model in ['bernoulli', 'ordinal']:
            sufficient_statistics['log-likelihood'] = self.compute_individual_attachment_tensorized(data,
                                                                                                    individual_parameters,
                                                                                                    attribute_type='MCMC')
        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):
        # Memoryless part of the algorithm

        # unlink model parameters from updates in realizations!
        realizations = realizations.clone_realizations()

        self.parameters['g'] = realizations['g'].tensor_realizations
        xi = realizations['xi'].tensor_realizations
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                self.parameters['deltas'] = realizations['deltas'].tensor_realizations
            else:
                for feat in self.ordinal_infos["features"]:
                    self.parameters['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

        param_ind = self.get_param_from_real(realizations)
        if self.noise_model in ['bernoulli', 'ordinal']:
            self.parameters['log-likelihood'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                              attribute_type='MCMC').sum()
        else:
            self.parameters['noise_std'] = NoiseModel.rmse_model(self, data, param_ind, attribute_type='MCMC')

    def update_model_parameters_normal(self, data, suff_stats):
        # Stochastic sufficient statistics used to update the parameters of the model

        self.parameters['g'] = suff_stats['g']

        tau_mean = self.parameters['tau_mean']
        tau_var_updt = torch.mean(suff_stats['tau_sqrd']) - 2. * tau_mean * torch.mean(suff_stats['tau'])
        tau_var = tau_var_updt + tau_mean ** 2
        self.parameters['tau_std'] = self._compute_std_from_var(tau_var, varname='tau_std')
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_var_updt = torch.mean(suff_stats['xi_sqrd']) - 2. * xi_mean * torch.mean(suff_stats['xi'])
        xi_var = xi_var_updt + xi_mean ** 2
        self.parameters['xi_std'] = self._compute_std_from_var(xi_var, varname='xi_std')
        self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                self.parameters['deltas'] = suff_stats['deltas']
            else:
                for feat in self.ordinal_infos["features"]:
                    self.parameters['deltas_'+feat["name"]] = suff_stats['deltas_'+feat["name"]]

        if self.noise_model in ['bernoulli', 'ordinal']:
            self.parameters['log-likelihood'] = suff_stats['log-likelihood'].sum()

        elif 'scalar' in self.noise_model:
            # scalar noise (same for all features)
            S1 = data.L2_norm
            S2 = suff_stats['obs_x_reconstruction'].sum()
            S3 = suff_stats['reconstruction_x_reconstruction'].sum()

            noise_var = (S1 - 2. * S2 + S3) / data.n_observations
            self.parameters['noise_std'] = self._compute_std_from_var(noise_var, varname='noise_std')
        else:
            # keep feature dependence on feature to update diagonal noise (1 free param per feature)
            S1 = data.L2_norm_per_ft
            S2 = suff_stats['obs_x_reconstruction'].sum(dim=(0, 1))
            S3 = suff_stats['reconstruction_x_reconstruction'].sum(dim=(0, 1))

            # tensor 1D, shape (dimension,)
            noise_var = (S1 - 2. * S2 + S3) / data.n_observations_per_ft.float()
            self.parameters['noise_std'] = self._compute_std_from_var(noise_var, varname='noise_std')


    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        ## Individual variables
        tau_infos = {
            "name": "tau",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        xi_infos = {
            "name": "xi",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.noise_model == 'ordinal':
            if self.ordinal_infos['batch_deltas']:
                # For each feature : create a sampler for deltas of size (nb_levels_of_the_feature)
                max_level = self.ordinal_infos["max_level"]
                deltas_infos = {
                    "name": "deltas",
                    "shape": torch.Size([self.dimension, max_level - 1]),
                    "type": "population",
                    "rv_type": "multigaussian",
                    "scale": .5,
                    "mask": self.ordinal_infos["mask"][0,0,:,1:], # cut the zero level
                }
                variables_infos['deltas'] = deltas_infos
            else:
                # Instead of a sampler for each feature, sample deltas for all features in one sampler class
                for feat in self.ordinal_infos["features"]:
                    deltas_infos = {
                        "name": "deltas_"+feat["name"],
                        "shape": torch.Size([feat["nb_levels"] - 1]),
                        "type": "population",
                        "rv_type": "gaussian",
                        "scale": .5,
                    }
                    variables_infos['deltas_'+feat["name"]] = deltas_infos

        return variables_infos

# document some methods (we cannot decorate them at method creation since they are not yet decorated from `doc_with_super`)
doc_with_(UnivariateModel.compute_individual_tensorized_linear,
          UnivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(UnivariateModel.compute_individual_tensorized_logistic,
          UnivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (logistic)'})

doc_with_(UnivariateModel.compute_jacobian_tensorized_linear,
          UnivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(UnivariateModel.compute_jacobian_tensorized_logistic,
          UnivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (logistic)'})

doc_with_(UnivariateModel.compute_individual_ages_from_biomarker_values_tensorized_logistic,
          UnivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
          mapping={'the model': 'the model (logistic)'})
