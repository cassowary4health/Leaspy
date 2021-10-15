######################## Tests for variability on directions ################################
# Not functional

import numpy as np
import math
import torch


def vectoangle(s):
    """
    Parameters:
    -s an normalized vector of dimension d
    return:
    - theta the angles vector of dimension d-1

    """
    if len(s.shape) == 1:
        d = len(s)
        L = torch.zeros(d - 1)
        L[0] = torch.acos(s[0])
        S = 1.0
        for i in range(1, d - 1):
            S = torch.sin(L[i - 1]) * S
            L[i] = torch.acos(s[i] / S)
        return L
    elif len(s.shape) == 2:
        d = s.shape[1]
        N = s.shape[0]
        L = torch.zeros(N, d - 1)
        L[:, 0] = torch.acos(s[:, 0])
        S = torch.ones(N)
        for i in range(1, d - 1):
            S = torch.sin(L[:, i - 1]) * S
            L[:, i] = torch.acos(s[:, i] / S)
        return L
    else:
        raise ValueError


def angletovec(theta):
    """
    Parameters:
    -s an normalized vector of dimension d
    - theta the angles vector of dimension d-1
    return:
    -s an normalized vector of dimension d
    """
    if len(theta.shape) == 1:
        dminusone = len(theta)
        L = torch.zeros(dminusone + 1)
        L[0] = torch.cos(theta[0])
        S = 1.0

        for i in range(1, dminusone):
            S = torch.sin(theta[i - 1]) * S
            L[i] = torch.cos(theta[i]) * S
        S = torch.sin(theta[dminusone - 1]) * S
        L[dminusone] = S
        return L
    elif len(theta.shape) == 2:
        N = theta.shape[0]
        dminusone = theta.shape[1]
        L = torch.zeros(N, dminusone + 1)
        L[:, 0] = torch.cos(theta[:, 0])
        S = torch.ones(N)

        for i in range(1, dminusone):
            S = torch.sin(theta[:, i - 1]) * S
            L[:, i] = torch.cos(theta[:, i]) * S
        S = torch.sin(theta[:, dminusone - 1]) * S
        L[:, dminusone] = S
        return L
    else:
        raise ValueError


def rotation(w, thetaplus):
    if len(w.shape) == 1:
        norm = torch.norm(w)
        wnorm = w.clone()
        wnorm = w / norm
        repangle = vectoangle(wnorm)
        rep = repangle + thetaplus
        wnew = angletovec(rep) * norm
        return wnew
    elif len(w.shape) == 2:
        norm = torch.norm(w, dim=1)
        wnorm = w.clone()
        wnorm[norm > 10 ** (-5)] = w[norm > 10 ** (-5)] / norm.unsqueeze(-1)[norm > 10 ** (-5)]

        repangle = vectoangle(wnorm)

        rep = repangle + thetaplus
        wnew = angletovec(rep) * norm.unsqueeze(-1)

        return wnew
    else:
        raise ValueError

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_factory import AttributesFactory

from .utils.initialization.model_initialization import initialize_parameters
from leaspy import __version__
import json

class LinearVari(AbstractMultivariateModel):
    """
    Logistic model for multiple variables of interest.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.neg=False
        self.source_dimension_direction=None
        self.parameters["Param"] = None
        self.components=None
        self.MCMC_toolbox['priors']['Param_std'] = None  # Value, Coef

    def save(self, path, with_mixing_matrix=True, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path: str
            Path to store the model's parameters.
        with_mixing_matrix: bool (default True)
            Save the mixing matrix in the exported file in its 'parameters' section.
            <!> It is not a real parameter and its value will be overwritten at model loading
                (orthonormal basis is recomputed from other "true" parameters and mixing matrix is then deduced from this orthonormal basis and the betas)!
            It was integrated historically because it is used for convenience in browser webtool and only there...
        **kwargs
            Keyword arguments for json.dump method.
        """
        model_parameters_save = self.parameters.copy()

        if with_mixing_matrix:
            model_parameters_save['mixing_matrix'] = self.attributes.mixing_matrix

        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()

        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'source_dimension_direction':self.source_dimension_direction,
            'loss': self.loss,
            'parameters': model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,source_dimension_direction=self.source_dimension_direction)
        self.attributes.update(['all'], self.parameters)

    def initialize(self, dataset, method="default"):
        self.dimension = dataset.dimension
        self.features = dataset.headers

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))

        self.parameters = initialize_parameters(self, dataset, method)
        

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,self.source_dimension_direction)
        self.attributes.update(['all'], self.parameters)
        self.is_initialized = True

   

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):

        #à changer, étape changement de direction

        
        # Population parameters
        positions, param, mixing_matrix = self._get_attributes(attribute_type)
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)
        sources=ind_parameters['sources']
        
        if self.source_dimension_direction != 0 :
            V=self.components
            if len(sources.shape)>1:
                theta_sources = sources[:,self.source_dimension:]

                thetai = theta_sources.matmul(V)
                
                param=rotation(param,thetai)
                
                
                param = param.unsqueeze(1) 
            else:
                theta_sources = sources[self.source_dimension:]
                thetai = theta_sources.matmul(V)
                
                param=rotation(param,thetai)
                
                param = param.reshape(1, 1, -1)
            
            positions = positions.reshape(1, 1, -1) # same
            reparametrized_time = reparametrized_time.unsqueeze(-1)
            
             # Computation
            LL = param * reparametrized_time + positions

            if self.source_dimension != 0:
                if len(sources.shape)>1:
                    sources_w = sources[:,:self.source_dimension]
                else:
                    sources_w = sources[:self.source_dimension]

                wi = sources_w.matmul(mixing_matrix.t())
                
                wi=rotation(wi,thetai)
                
                

                LL += wi.unsqueeze(-2)
        

        else:
            param = param.reshape(1, 1, -1) # not needed in fact (automatic broadcasting on last dimension)
            positions = positions.reshape(1, 1, -1) # same
            reparametrized_time = reparametrized_time.unsqueeze(-1)

             # Computation
            LL = param * reparametrized_time + positions

            if self.source_dimension != 0:
                sources = ind_parameters['sources']
                wi = sources.matmul(mixing_matrix.t())

                LL += wi.unsqueeze(-2)

        return LL # (n_individuals, n_timepoints, n_features)

    def compute_individual_tensorized_mixed(self, timepoints, ind_parameters, attribute_type=None):
        raise NotImplementedError

    
    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        raise NotImplementedError
  
    """
    à changer plus tard, permet d'accélérer la détermination des paramètres personalisés
    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name == 'logistic':
            return self.compute_jacobian_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name == 'linear':
            return self.compute_jacobian_tensorized_linear(timepoints, ind_parameters, attribute_type)
        elif self.name == 'mixed_linear-logistic':
            return self.compute_jacobian_tensorized_mixed(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError("Mutivariate model > Compute jacobian tensorized")
   
"""
    def compute_jacobian_tensorized_mixed(self, timepoints, ind_parameters, attribute_type=None):
        raise NotImplementedError


   
    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'Param_std': 0.01, 'betas_std': 0.01},
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,source_dimension_direction=self.source_dimension_direction)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # TODO maybe not here
        # Initialize priors
        self.MCMC_toolbox['priors']['Param_mean'] = self.parameters['Param'].clone()
        self.MCMC_toolbox['priors']['s_Param'] = 0.1

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('Param', 'all')):
            values['Param'] = realizations['Param'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        #On recentre les ci de cette manière en vue de l'argument de l'exponentielle au dénominateur
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['Param'].tensor_realizations[:self.dimension] = realizations['Param'].tensor_realizations[:self.dimension] + mean_xi
        #On modifie les self.dimension premiers termes car c'est les taux d'infection
        self.update_MCMC_toolbox(['all'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations,
            'Param': realizations['Param'].tensor_realizations,
            'tau': realizations['tau'].tensor_realizations,
            'tau_sqrd': torch.pow(realizations['tau'].tensor_realizations, 2),
            'xi': realizations['xi'].tensor_realizations,
            'xi_sqrd': torch.pow(realizations['xi'].tensor_realizations, 2)
        }
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations

        ind_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 ind_parameters,
                                                                 attribute_type='MCMC')

        data_reconstruction *= data.mask.float()  # speed-up computations

        norm_1 = data.values * data_reconstruction  # * data.mask.float()
        norm_2 = data_reconstruction * data_reconstruction  # * data.mask.float()

        sufficient_statistics['obs_x_reconstruction'] = norm_1  # .sum(dim=2) # no sum on features...
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2  # .sum(dim=2) # no sum on features...

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters,
                                                                                                  attribute_type="MCMC")

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations

        if self.MCMC_toolbox['priors']['Param_mean'] is not None:
            Param_mean = self.MCMC_toolbox['priors']['Param_mean']
            Param_emp = realizations['Param'].tensor_realizations
            s_Param = self.MCMC_toolbox['priors']['s_Param']
            sigma_Param = self.MCMC_toolbox['priors']['Param_std']
            self.parameters['Param'] = (1 / (1 / (s_Param ** 2) + 1 / (sigma_Param ** 2))) * (
                        Param_emp / (sigma_Param ** 2) + Param_mean / (s_Param ** 2))
        else:
            self.parameters['Param'] = realizations['Param'].tensor_realizations

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations
        xi = realizations['xi'].tensor_realizations
        # self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        # TODO : Why is it MCMC-SAEM? SHouldn't it be computed with the parameters?
        if 'diag_noise' in self.loss:
            squared_diff_per_ft = self.compute_sum_squared_per_ft_tensorized(data, param_ind, attribute_type='MCMC').sum(dim=0)  # sum on individuals
            self.parameters['noise_std'] = torch.sqrt(squared_diff_per_ft / data.n_observations_per_ft.float())
        else:
            squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type='MCMC').sum()  # sum on individuals
            self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)

        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                            attribute_type="MCMC").sum()

        # TODO : This is just for debugging of linear
        # data_reconstruction = self.compute_individual_tensorized(data.timepoints,
        #                                                         self.get_param_from_real(realizations),
        #                                                         attribute_type='MCMC')
        # norm_0 = data.values * data.values * data.mask.float()
        # norm_1 = data.values * data_reconstruction * data.mask.float()
        # norm_2 = data_reconstruction * data_reconstruction * data.mask.float()
        # S1 = torch.sum(torch.sum(norm_0, dim=2))
        # S2 = torch.sum(torch.sum(norm_1, dim=2))
        # S3 = torch.sum(torch.sum(norm_2, dim=2))

        # print("During burn-in : ", torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits)),
        #       torch.sqrt(squared_diff / (data.n_visits * data.dimension)))

        # Stochastic sufficient statistics used to update the parameters of the model

    def update_model_parameters_normal(self, data, suff_stats):
        # TODO with Raphael : check the SS, especially the issue with mean(xi) and v_k
        # TODO : 1. Learn the mean of xi and v_k
        # TODO : 2. Set the mean of xi to 0 and add it to the mean of V_k
        self.parameters['g'] = suff_stats['g']
        self.parameters['Param'] = suff_stats['Param']
        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']

        tau_mean = self.parameters['tau_mean'].clone()
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(tau_std_updt + self.parameters['tau_mean'] ** 2)
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
        self.parameters['xi_std'] = torch.sqrt(xi_std_updt + self.parameters['xi_mean'] ** 2)
        # self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        if 'diag_noise' in self.loss:
            # keep feature dependence on feature to update diagonal noise (1 free param per feature)
            S1 = data.L2_norm_per_ft
            S2 = suff_stats['obs_x_reconstruction'].sum(dim=(0, 1))
            S3 = suff_stats['reconstruction_x_reconstruction'].sum(dim=(0, 1))

            self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations_per_ft.float())
            # tensor 1D, shape (dimension,)
        else: # scalar noise (same for all features)
            S1 = data.L2_norm
            S2 = suff_stats['obs_x_reconstruction'].sum()
            S3 = suff_stats['reconstruction_x_reconstruction'].sum()

            self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations)

        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()

        # print("After burn-in : ", torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits)))

    ###################################
    ### Random Variable Information ###
    ###################################

    def random_variable_informations(self):

        # --- Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        Param_infos = {
            "name": "Param",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension-1, self.source_dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        # --- Individual variables
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

        sources_infos = {
            "name": "sources",
            "shape": torch.Size([self.source_dimension+self.source_dimension_direction]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "Param": Param_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos
        return variables_infos

################
# Initialization
################

def initialize_linear_vari(model, dataset, method):
    """
    Initialize the linear model's group parameters.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g', 'v0', 'betas', 'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std' and 'noise_std'.
    """

    if model.source_dimension_direction is None:
        model.source_dimension_direction = model.source_dimension
    sum_ages = torch.sum(dataset.timepoints).item()
    nb_nonzeros = (dataset.timepoints != 0).sum()

    t0 = float(sum_ages) / float(nb_nonzeros)

    df = dataset.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)

    positions, velocities = [[] for _ in range(model.dimension)], [[] for _ in range(model.dimension)]

    for idx in dataset.indices:
        indiv_df = df.loc[idx]
        ages = indiv_df.index.values
        features = indiv_df.values

        if len(ages) == 1:
            continue

        for dim in range(model.dimension):

            ages_list, feature_list = [], []
            for i, f in enumerate(features[:, dim]):
                if f == f:
                    feature_list.append(f)
                    ages_list.append(ages[i])

            if len(ages_list) < 4:
                break
            else:
                slope, intercept, _, _, _ = stats.linregress(ages_list, feature_list)

                value = intercept + t0 * slope

                velocities[dim].append(slope)
                positions[dim].append(value)

    positions = [torch.tensor(_) for _ in positions]
    positions = torch.tensor([torch.mean(_) for _ in positions], dtype=torch.float32)
    velocities = torch.tensor(velocities)

    velocities = velocities.permute(1, 0)
    velocities0 = torch.mean(velocities, axis=0)
    print("velo")
    print(torch.isnan(velocities).any())
    velocitiesnorm = velocities / torch.norm(velocities, dim=1).unsqueeze(-1)
    print("velonorm")
    print(torch.isnan(velocitiesnorm).any())

    angles = [vectoangle(velocitiesnorm[i]).tolist() for i in range(len(velocitiesnorm))]

    angles_to_pca = np.array(angles)
    print("velonorm")
    print(np.isnan(angles_to_pca).any())
    pca = PCA(n_components=model.source_dimension_direction)
    pca.fit(angles_to_pca)
    print(pca.singular_values_)
    V = torch.tensor(list(pca.components_), dtype=torch.float32)
    print(V)

    if 'univariate' in model.name:
        if (velocities0 <= 0).item():
            warnings.warn(
                "Individual linear regressions made at initialization has a mean slope which is negative: not properly handled in case of an univariate linear model...")
            xi_mean = torch.tensor(-3.)  # default...
        else:
            xi_mean = torch.log(velocities0).squeeze()

        parameters = {
            'g': positions.squeeze(),
            'tau_mean': torch.tensor(t0), 'tau_std': torch.tensor(tau_std),
            'xi_mean': xi_mean, 'xi_std': torch.tensor(xi_std),
            'noise_std': torch.tensor(noise_std, dtype=torch.float32)
        }
    else:
        model.components = V
        parameters = {
            'g': positions,
            'Param': velocities0,
            'betas': torch.zeros((model.dimension - 1, model.source_dimension)),
            'tau_mean': torch.tensor(t0), 'tau_std': torch.tensor(tau_std),
            'xi_mean': torch.tensor(0.), 'xi_std': torch.tensor(xi_std),
            'sources_mean': torch.tensor(0.), 'sources_std': torch.tensor(sources_std),
            'noise_std': torch.tensor([noise_std], dtype=torch.float32)
        }

    return parameters

###########
#Attributes
###########

import torch

from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLinearVari(AttributesAbstract):
    """
    Contains the common attributes & methods to update the logistic_asymp model's attributes.

    Attributes
    ----------
    dimension: `int`
    source_dimension: `int`
    betas: `torch.Tensor` (default None)
    positions: `torch.Tensor` (default None)
        positions = exp(realizations['g']) such that p0 = 1 / (1+exp(g))
    mixing_matrix: `torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i
    orthonormal_basis: `torch.Tensor` (default None)
    velocities: `torch.Tensor` (default None)
    name: `str` (default 'logistic')
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: `tuple` [`str`] (default ('all', 'g', 'v0', 'betas') )
        Contains the available parameters to update. Different models have different parameters.

    Methods
    -------
    get_attributes()
        Returns the following attributes: ``positions``, ``deltas`` & ``mixing_matrix``.
    update(names_of_changed_values, values)
        Update model group average parameter(s).
    """

    def __init__(self, name, dimension, source_dimension, source_dimension_direction):
        """
        Instantiate a AttributesLogistic class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        self.Param = None

        super().__init__(name, dimension, source_dimension)
        self.update_possibilities = ('all', 'g', 'Param', 'betas')

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a basis orthogonal to velocities v0 for the inner product
        implied by the metric..
        """
        dgamma_t0 = self.Param
        self._compute_Q(dgamma_t0)

    def get_attributes(self):  # à changer
        """
        Returns the following attributes: ``positions``, ``Param`` & ``mixing_matrix``.

        Returns
        -------
        - positions: `torch.Tensor`
        - velocities: `torch.Tensor`
        - mixing_matrix: `torch.Tensor`
        """

        return self.positions, self.Param, self.mixing_matrix

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values: `list` [`str`]
            Must be one of - "all", "g", "v0", "betas". Raise an error otherwise.
            "g" correspond to the attribute ``positions``.
            "v0" correspond to the attribute ``velocities``.
        values: `dict` [`str`, `torch.Tensor`]
            New values used to update the model's group average parameters
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_positions = False
        compute_Param = False

        if 'all' in names_of_changed_values:
            names_of_changed_values = self.update_possibilities  # make all possible updates

        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'g' in names_of_changed_values:
            compute_positions = True
        if ('Param' in names_of_changed_values) or ('xi_mean' in names_of_changed_values):
            compute_Param = True

        if compute_betas:
            self._compute_betas(values)
        if compute_positions:
            self._compute_positions(values)

        if compute_Param:
            self._compute_Param(values)

        # TODO : Check if the condition is enough
        if self.has_sources and (compute_positions or compute_Param):
            self._compute_orthonormal_basis()
        if self.has_sources and (compute_positions or compute_Param or compute_betas):
            self._compute_mixing_matrix()

    def _check_names(self, names_of_changed_values):
        """
        Check if the name of the parameter(s) to update are in the possibilities allowed by the model.

        Parameters
        ----------
        names_of_changed_values: `list` [`str`]

        Raises
        -------
        ValueError
        """
        unknown_update_possibilities = set(names_of_changed_values).difference(self.update_possibilities)
        if len(unknown_update_possibilities) > 0:
            raise ValueError(f"{unknown_update_possibilities} not in the attributes that can be updated")

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.positions = torch.exp(values['g'])  # on a échantilloné suivant une loi normale

    def _compute_Param(self, values):
        """
        Update the attribute ``Param``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """

        self.Param = torch.exp(values['Param'])

    # overwrite les compute_positions get_attributes
