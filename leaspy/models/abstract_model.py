import math
import warnings

import torch

from leaspy.utils.realizations.collection_realization import CollectionRealization
from leaspy.utils.realizations.realization import Realization

TWO_PI = 2 * math.pi


# TODO: Check & complete docstrings
class AbstractModel:
    """
    AbstractModel class contains the common attributes & methods of the different models.

    Attributes
    ----------
    is_initialized: bool
        Indicates if the model is initialized.
    features: list [str]
        Contains the features' name on which the model has been calibrated.
    name: str
        The model's name.
    parameters: dict
        Contains the model's parameters.
    source_dimension: int, (default 0)
        The number of sources used.
    individual_parameters_posterior_distribution: dict [str, torch.Tensor]
        Contains the individual parameters' mean and covariance matrix. Both of them are estimated at the end of the
        calibration. The variable are stored in the following order: xi, tau then sources.

    Methods
    -------
    compute_individual_attachment_tensorized_mcmc(data, realizations)
        Compute attachment of all subjects? One subject? One visit?
    compute_sum_squared_tensorized(data, param_ind, attribute_type=None)
        Compute the square of the residuals. (?) from one subject? Several subjects? All subject?
    get_individual_variable_name()
        Return list of names of the individual variables from the model.
    load_parameters(parameters)
        Instantiate or update the model's parameters.
     set_individual_parameters_distribution(self, individual_parameters)
        Set the attribute ``individual_parameters_distributions`` from a dictionary of ``individual_parameters``.
    compute_multivariate_gaussian_posterior_regularity(self, value)
        Given the individual parameter of a subject, compute its regularity compared to the posterior distribution
        of the individual parameters.
    """

    def __init__(self, name):
        self.is_initialized = False
        self.name = name
        self.features = None
        self.parameters = None
        self.source_dimension = 0
        self.attributes = None

        self._last_realisations = None
        self._univariate_gaussian_distribution = torch.distributions.normal.Normal(loc=0., scale=1.)
        self.individual_parameters_posterior_distribution = None

        self._sources_conditional_posterior_covariance = None
        self._sources_conditional_posterior_covariance_inverse = None

        self._omegas_posterior_mean = None
        self._omegas_posterior_covariance = None
        self._omegas_posterior_covariance_inverse = None

    def load_parameters(self, parameters):
        """
        Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters: dict
            Contains the model's parameters
        """
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = parameters[k]

    def load_posterior_distribution(self, posterior_distribution):
        """
        Create the attribute ``individual_parameters_posterior_distribution`` which is a torch `MultivariateNormal`
        class object from its mean and covariance matrix contained in the input.

        Parameters
        ----------
        posterior_distribution: dict [str, torch.Tensor]
            Contains the mean and covariance matrix of the posterior distribution.
        """
        if all([val is not None for val in posterior_distribution.values()]):
            self.individual_parameters_posterior_distribution = \
                torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=posterior_distribution['mean'],
                    covariance_matrix=posterior_distribution['covariance']
                )
            if self.name != 'univariate':
                self.set_sources_distribution()
                self.set_omegas_distribution()
        else:
            warnings.warn('The posterior distribution could not have been loaded from this  model file!', stacklevel=4)

    def load_hyperparameters(self, hyperparameters):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def get_individual_variable_name(self):
        """
        Return list of names of the individual variables from the model.

        Returns
        -------
        individual_variable_name : `list` [str]
            Contains the individual variables' names
        """
        individual_variable_name = []

        infos = self.random_variable_informations()  # overloaded for each model
        for name, info in infos.items():
            if info['type'] == 'individual':
                individual_variable_name.append(name)

        return individual_variable_name

    def compute_sum_squared_tensorized(self, data, param_ind, attribute_type=None):
        """
        Compute for each subject the sum of its squared residuals. The sum is on every scores and every visits.

        Parameters
        ----------
        data : leaspy.inputs.data.dataset.Dataset
            Contains the data of the subjects, in particular the subjects' values, time-points and the mask.
        param_ind : dict [str, torch.Tensor]
            Contain the individual parameters. ``xi`` and ``tau`` have shape of ``(n_subjects, 1)``. If the model is
            not `univariate`, ``sources`` has shape ``(n_subjects, n_sources)``.
        attribute_type : str, optional (default None)
            The attribute's type.

        Returns
        -------
        torch.Tensor, shape = (n_subjects,)
            Contain for each subject the sum of its squared residuals. The sum is on every scores and every visits.
        """
        res: torch.FloatTensor = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        return torch.sum((res * data.mask.float() - data.values) ** 2, dim=(1, 2))

    def audit_individual_parameters(self, ips):
        """
        Perform various consistency and compatibility (with current model) checks
        on an individual parameters dict and outputs qualified information about it.

        Parameters
        ----------
        ips: dict
            Contains some untrusted individual parameters.
            If representing only one individual (in a multivariate model) it could be:
                {'tau':0.1, 'xi':-0.3, 'sources':[0.1,...]}
            Or for multiple individuals:
                {'tau':[0.1,0.2,...], 'xi':[-0.3,0.2,...], 'sources':[[0.1,...],[0,...],...]}
            In particular, a sources vector (if present) should always be a array_like, even if it is 1D

        Returns
        -------
        ips_info: dict
            'nb_inds': number of individuals present (int >= 0)
            'tensorized_ips': tensorized version of individual parameters
            'tensorized_ips_gen': generator providing for all individuals present (ordered as is)
                their own tensorized individual parameters

        Raises
        ------
        ValueError: if any of the consistency/compatibility checks fail
        """

        def is_array_like(v):
            # abc.Collection is useless here because set, np.array(scalar) or torch.tensor(scalar)
            # are abc.Collection but are not array_like in numpy/torch sense or have no len()
            try:
                len(v)  # exclude np.array(scalar) or torch.tensor(scalar)
                return hasattr(v, '__getitem__')  # exclude set
            except TypeError:
                return False

        # Model supports and needs sources?
        has_sources = self.name != 'univariate'

        # Check parameters names
        expected_parameters = set(['xi', 'tau'] + int(has_sources) * ['sources'])
        given_parameters = set(ips.keys())
        symmetric_diff = expected_parameters.symmetric_difference(given_parameters)
        if len(symmetric_diff) > 0:
            raise ValueError('Individual parameters dict provided {} is not compatible for {} model. ' \
                             'The expected individual parameters are {}.'. \
                             format(given_parameters, self.name, expected_parameters))

        # Check number of individuals present (with low constraints on shapes)
        ips_is_array_like = {k: is_array_like(v) for k, v in ips.items()}
        ips_size = {k: len(v) if ips_is_array_like[k] else 1 for k, v in ips.items()}

        if has_sources:
            s = ips['sources']

            if not ips_is_array_like['sources']:
                raise ValueError('Sources must be an array_like but {} was provided.'. \
                                 format(s))

            tau_xi_scalars = all(ips_size[k] == 1 for k in ['tau', 'xi'])
            if tau_xi_scalars and (ips_size['sources'] > 1):
                # is 'sources' not a nested array? (allowed iff tau & xi are scalars)
                if not is_array_like(s[0]):
                    # then update sources size (1D vector representing only 1 individual)
                    ips_size['sources'] = 1

            # TODO? check source dimension compatibility?

        uniq_sizes = set(ips_size.values())
        if len(uniq_sizes) != 1:
            raise ValueError('Individual parameters sizes are not compatible together. ' \
                             'Sizes are {}.'.format(ips_size))

        # number of individuals present
        n_inds = uniq_sizes.pop()

        # properly choose unsqueezing dimension when tensorizing array_like (useful for sources)
        unsqueeze_dim = -1  # [1,2] => [[1],[2]] (expected for 2 individuals / 1D sources)
        if n_inds == 1:
            unsqueeze_dim = 0  # [1,2] => [[1,2]] (expected for 1 individual / 2D sources)

        # tensorized (2D) version of ips
        t_ips = {k: self._tensorize_2D(v, unsqueeze_dim=unsqueeze_dim) for k, v in ips.items()}

        # construct output
        return {
            'nb_inds': n_inds,
            'tensorized_ips': t_ips,
            'tensorized_ips_gen': ({k: v[i, :].unsqueeze(0) for k, v in t_ips.items()} for i in range(n_inds))
        }

    @staticmethod
    def _tensorize_2D(x, unsqueeze_dim, dtype=torch.float32):
        """
        Helper to convert a scalar or array_like into an, at least 2D, dtype tensor.

        Parameters
        ----------
        x: scalar or array_like
            Element to be tensorized.
        unsqueeze_dim: 0 or -1
            dimension to be unsqueezed; meaningful for 1D array-like only.

        Examples
        --------
        >>> import torch
        >>> from leaspy.models.abstract_model import AbstractModel
        >>> model = AbstractModel()
        >>> model._tensorize_2D([1, 2], 0) == torch.tensor([[1, 2]])
        >>> model._tensorize_2D([1, 2], -1) == torch.tensor([[1], [2])
        For scalar or vector of length 1 it has no matter
        """
        # convert to torch.Tensor if not the case
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=dtype)

        # convert dtype if needed
        if x.dtype != dtype:
            x = x.to(dtype)

        # if tensor is less than 2-dimensional add dimensions
        while x.dim() < 2:
            x = x.unsqueeze(dim=unsqueeze_dim)

        # postcondition: x.dim() >= 2
        return x

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_trajectory(self, timepoints, individual_parameters, *, skip_ips_checks=False):
        """
        Compute scores values at the given time-point(s) given a subject's individual parameters.

        Parameters
        ----------
        timepoints: scalar or array_like[scalar]
            Contains the age(s) of the subject.
        individual_parameters: dict
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.
        skip_ips_checks: bool, (default False)
            Flag to skip consistency/compatibility checks and tensorization
            of individual_parameters when it was done earlier (speed-up).

        Returns
        -------
        torch.Tensor
            Contains the subject's scores computed at the given age(s)
            Shape of tensor is (1, n_timepoints, n_features)
        """

        if not skip_ips_checks:
            # Perform checks on ips and gets tensorized version if needed
            ips_info = self.audit_individual_parameters(individual_parameters)
            n_inds = ips_info['nb_inds']
            individual_parameters = ips_info['tensorized_ips']

            if n_inds != 1:
                raise ValueError('Only one individual computation may be performed at a time. ' \
                                 '{} was provided.'.format(n_inds))

        # Convert the timepoints (list of numbers, or single number) to a 2D torch tensor
        timepoints = self._tensorize_2D(timepoints, unsqueeze_dim=0)  # 1 individual

        # Compute the individual trajectory
        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def compute_individual_tensorized(self, timepoints, individual_parameters, attribute_type=None):
        return NotImplementedError

    def compute_individual_attachment_tensorized_mcmc(self, data, realizations):
        """
        Compute attachment of the subjects for all visits.

        Parameters
        ----------
        data: leaspy.inputs.data.dataset.Dataset
            Contains the data of the subjects, in particular the subjects' scores, time-points and the mask.
        realizations: leaspy.utils.realizations.collection_realization.CollectionRealization
            Contains the subjects' individual parameters.

        Returns
        -------
        attachment : torch.Tensor
            The subjects' attachment.
        """
        param_ind = self.get_param_from_real(realizations)
        attachment = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type='MCMC')
        return attachment

    def compute_individual_attachment_tensorized(self, data, param_ind, attribute_type):
        """
        Compute attachment of the subjects for all visits.

        Parameters
        ----------
        data: leaspy.inputs.data.dataset.Dataset
            Contains the data of the subjects, in particular the subjects' time-points and the mask (?)
        param_ind: dict [str, torch.Tensor]
            Contains the subjects individual parameters.
        attribute_type: str

        Returns
        -------
        attachment: torch.Tensor
            The subjects' attachment.
        """
        res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        # res *= data.mask

        r1 = res * data.mask.float() - data.values  # r1.shape = (n_subjects, ??, n_features)
        # r1[1-data.mask] = 0.0 # Set nans to 0
        squared_sum = torch.sum(r1 * r1, dim=(1, 2))

        noise_var = self.parameters['noise_std'] * self.parameters['noise_std']
        attachment = 0.5 * (1. / noise_var) * squared_sum

        attachment += math.log(math.sqrt(TWO_PI * noise_var))
        return attachment

    def update_model_parameters(self, data, suff_stats, burn_in_phase=True):
        # Memoryless part of the algorithm
        if burn_in_phase:
            self.update_model_parameters_burn_in(data, suff_stats)
        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            self.update_model_parameters_normal(data, suff_stats)
        self.attributes.update(['all'], self.parameters)

    def update_model_parameters_burn_in(self, data, suff_stats):
        raise NotImplementedError

    def update_model_parameters_normal(self, data, suff_stats):
        raise NotImplementedError

    def get_population_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'population']

    def get_individual_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'individual']

    def __str__(self):
        output = "=== MODEL ===\n"
        for key in self.parameters.keys():
            # if type(self.parameters[key]) == float:
            #    output += "{} : {:.5f}\n".format(key, self.parameters[key])
            # else:
            output += "{} : {}\n".format(key, self.parameters[key])
        return output

    def compute_regularity_realization(self, realization):
        # Instantiate torch distribution
        if realization.variable_type == 'population':
            mean = self.parameters[realization.name]
            # TODO : Sure it is only MCMC_toolbox?
            std = self.MCMC_toolbox['priors']['{0}_std'.format(realization.name)]
        elif realization.variable_type == 'individual':
            mean = self.parameters["{0}_mean".format(realization.name)]
            std = self.parameters["{0}_std".format(realization.name)]
        else:
            raise ValueError("Variable type not known")

        return self.compute_regularity_variable(realization.tensor_realizations, mean, std)

    def compute_regularity_variable(self, value, mean, std):
        self._univariate_gaussian_distribution.loc = mean
        self._univariate_gaussian_distribution.scale = std
        return -self._univariate_gaussian_distribution.log_prob(value)

    def get_realization_object(self, n_individuals):
        # TODO : CollectionRealizations should probably get self.get_info_var rather than all self
        realizations = CollectionRealization()
        realizations.initialize(n_individuals, self)
        return realizations

    def random_variable_informations(self):
        raise NotImplementedError

    def smart_initialization_realizations(self, data, realizations):
        return realizations

    def _create_dictionary_of_population_realizations(self):
        pop_dictionary = {}
        for name_var, info_var in self.random_variable_informations().items():
            if info_var['type'] != "population":
                continue
            real = Realization.from_tensor(name_var, info_var['shape'], info_var['type'], self.parameters[name_var])
            pop_dictionary[name_var] = real

        return pop_dictionary

    @staticmethod
    def time_reparametrization(timepoints, xi, tau):
        return torch.exp(xi) * (timepoints - tau)

    def get_param_from_real(self, realizations):

        individual_parameters = dict.fromkeys(self.get_individual_variable_name())

        for variable_ind in self.get_individual_variable_name():
            if variable_ind == "sources" and self.source_dimension == 0:
                individual_parameters[variable_ind] = None
            else:
                individual_parameters[variable_ind] = realizations[variable_ind].tensor_realizations

        return individual_parameters

    def set_individual_parameters_distribution(self, individual_parameters):
        """
        Set the attribute ``individual_parameters_posterior_distribution`` from a dictionary of
        ``individual_parameters``.

        Parameters
        ----------
        individual_parameters : dict [str, torch.Tensor]
            Contains log-acceleration 'xi', time-shifts 'tau' (& 'sources' if multivariate).
            Tau and xi have shape = (n_subjects, 1) and sources have shape = (n_subjects, n_sources).
        """
        # ------ Impose the order on tau, xi, sources
        tensor_list = [individual_parameters['tau'], individual_parameters['xi']]
        if self.name != 'univariate':
            tensor_list += [individual_parameters['sources']]
        ind_param = torch.cat(tensor_list, dim=-1)

        # ------ Compute mean and covariance matrix
        ind_param_mean = torch.mean(ind_param, dim=0)
        ind_param_cov = ind_param - ind_param_mean[None, :]
        ind_param_cov = 1 / (ind_param_cov.size(0) - 1) * ind_param_cov.t() @ ind_param_cov

        # ------ Impose xi & tau independent to each other & independent to sources
        # ind_param_cov[0, 1:] = 0.
        # ind_param_cov[1:, 0] = 0.
        # ind_param_cov[1, 2:] = 0.
        # ind_param_cov[2:, 1] = 0.

        self.individual_parameters_posterior_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=ind_param_mean, covariance_matrix=ind_param_cov)

    def set_sources_distribution(self):
        """
        Set the attribute ``sources_posterior_conditional_distribution`` from the attribute
        ``individual_parameters_posterior_distribution``.
        """
        if self.name != 'univariate':
            if self.individual_parameters_posterior_distribution is not None:
                sigma_11 = self.individual_parameters_posterior_distribution.covariance_matrix.narrow(
                    0, 0, 2).narrow(1, 0, 2)
                sigma_22 = self.individual_parameters_posterior_distribution.covariance_matrix.narrow(
                    0, 2, self.source_dimension).narrow(1, 2, self.source_dimension)
                sigma_12 = self.individual_parameters_posterior_distribution.covariance_matrix.narrow(
                    0, 2, self.source_dimension).narrow(1, 0, 2)

                # ------ Compute the conditional covariance matrix of the sources knowing (tau, xi)
                self._sources_conditional_posterior_covariance = sigma_22 - sigma_12 @ sigma_11.inverse() \
                                                                 @ sigma_12.transpose(0, -1)
                self._sources_conditional_posterior_covariance_inverse = \
                    self._sources_conditional_posterior_covariance.inverse()
            else:
                raise ValueError('The attribute "individual_parameters_posterior_distribution" of your model is None! '
                                 'First you need to calibrate this model.')

    def set_omegas_distribution(self):
        """
        Set the attribute ``_omegas_posterior_conditional_mean`` & ``_omegas_posterior_conditional_covariance``
        from the attribute ``individual_parameters_posterior_distribution``.
        """
        if self.name != 'univariate':
            if self.individual_parameters_posterior_distribution is not None:
                # ------ Get sources posterior mean & covariance matrix
                mu_sources = self.individual_parameters_posterior_distribution.loc[2:]
                sigma_sources = self.individual_parameters_posterior_distribution.covariance_matrix.narrow(
                    0, 2, self.source_dimension).narrow(1, 2, self.source_dimension)

                # ------ Compute the omegas posterior mean & covariance matrix
                self._omegas_posterior_mean = self.attributes.mixing_matrix @ mu_sources
                self._omegas_posterior_covariance = self.attributes.mixing_matrix @ sigma_sources @ \
                                                    self.attributes.mixing_matrix.t()
            else:
                raise ValueError('The attribute "individual_parameters_posterior_distribution" of your model is None! '
                                 'First you need to calibrate this model.')

    def set_omegas_posterior_covariance_inverse(self, omegas_regularity_factor):
        """
        Compute the space-shifts posterior covariance inverse matrix with the given regularization factor. Indeed,
        as soon as the number of sources is less than the number of scores, the space-shifts are not linearly
        independent.
        """
        n_omegas = self._omegas_posterior_mean.shape[0]
        regularized_covariance = self._omegas_posterior_covariance + omegas_regularity_factor * torch.eye(n_omegas)
        self._omegas_posterior_covariance_inverse = torch.inverse(regularized_covariance)

    def compute_multivariate_gaussian_posterior_regularity(self, value):
        """
        Given the individual parameter of a subject, compute its regularity assuming the individual parameters
        follow a multivariate normal distribution. We use the posterior distribution of the individual
        parameters of the cohort on which the model has been calibrated.

        Parameters
        ----------
        value: torch.Tensor, shape = (n_individual_parameters,)
            Contains the subject's individual parameters.

        Returns
        -------
        torch.Tensor
            The subject's regularity.
        """
        return -self.individual_parameters_posterior_distribution.log_prob(value)

    # TODO : Add non parametric method to compute regularity ? Ex with scipy.stats.gaussian_kde.integrate_kde
    # TODO : Pblm - it is very dependent of the selected bandwidth in the two distributions!

    def compute_multivariate_sources_regularity(self, tau_xi, sources):
        """
        Given the individual parameter of a subject, compute its the regularity of the `sources` assuming
        the individual parameters follow a multivariate normal distribution. We use the posterior distribution of the
        individual parameters of the cohort on which the model has been calibrated. To compute the regularity of
        the `sources`, the conditional mean and covariance matrix of the `sources` knowing `tau` and `xi` are used.

        Parameters
        ----------
        tau_xi: torch.Tensor, shape = (2,)
            Subject's (tau, xi).
        sources: torch.Tensor, shape = (n_sources,)
            Subject's sources.

        Returns
        -------
        torch.Tensor
            The subject's regularity.
        """
        mu_1 = self.individual_parameters_posterior_distribution.loc[:2]  # (tau, xi) mean
        mu_2 = self.individual_parameters_posterior_distribution.loc[2:]  # sources mean

        sigma_11 = self.individual_parameters_posterior_distribution.covariance_matrix.narrow(
            0, 0, 2).narrow(1, 0, 2)  # covariance matrix of (tau, xi)
        sigma_12 = self.individual_parameters_posterior_distribution.covariance_matrix.narrow(
            0, 2, self.source_dimension).narrow(1, 0, 2)  # covariance between (tau, xi) & the sources

        mean_cond = mu_2 + sigma_12 @ sigma_11.inverse() @ (tau_xi - mu_1)  # conditional sources mean knowing (tau, xi)

        diff = sources - mean_cond
        return diff @ self._sources_conditional_posterior_covariance_inverse @ diff

    def compute_multivariate_omegas_regularity(self, omegas):
        """
        Given the individual parameter of a subject, compute its the regularity of the `space-shifts`.

        Parameters
        ----------
        omegas: torch.Tensor, shape = (n_scores,)
            Subject's omegas (space-shifts).

        Returns
        -------
        torch.Tensor
            The subject's regularity.
        """
        diff = omegas - self._omegas_posterior_mean
        return diff @ self._omegas_posterior_covariance_inverse @ diff
