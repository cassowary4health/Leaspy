import torch

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes import AttributesFactory

from leaspy.utils.docs import doc_with_super, doc_with_

from leaspy import __version__
import json



initB = {"identity":lambda x:x,
"negidentity":lambda x:-x,
"logistic":lambda x:1./(1.+torch.exp(-x))}
import leaspy.models.utils.OptimB as OptimB

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete formulation depending on if linear, logistic, mixed log-lin, ...

@doc_with_super()
class LinearB(AbstractMultivariateModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.B= lambda x : x
        self.kernelsettings=None
        self.initB="identity"
        self.saveParam=[]
        self.parameters["v0"] = None
        
        self.saveB= []
        self.MCMC_toolbox['priors']['v0_std'] = None  # Value, Coef
    
    
 
    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)



   
    def initBlink(self):
        self.B=initB[self.initB]

    def reconstructionB(self):
        self.initBlink()
        for e in self.saveB:
            W,X_filtre=e
            W1=torch.tensor(W, dtype=torch.float32).clone().detach()
            X_filtre1=torch.tensor(X_filtre, dtype=torch.float32).clone().detach()
            FonctionTensor=OptimB.transformation_B_compose( X_filtre1,W1, self.kernelsettings,self.B)
            self.B=FonctionTensor

    
    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']
        expected_initB=("identity","negidentity","logistic")
        if 'kernelsettings' in hyperparameters.keys():
            self.kernelsettings = hyperparameters['kernelsettings']
        if 'init_b' in hyperparameters.keys():
            self.initB=hyperparameters['init_b']
            self.initBlink()
        if 'saveparam' in hyperparameters.keys():
            self.saveParam=hyperparameters['saveparam']
            
        if 'save_b' in hyperparameters.keys():
            
            L=[]
            for e in hyperparameters['save_b']:
                W,X=e
                L.append((torch.tensor(W),torch.tensor(X)))
            self.saveB = L
            self.reconstructionB()

    

        if 'B' in hyperparameters.keys():
            if 'initB' not in hyperparameters.keys():
                print("don't forget to define model.initB in order to save your parameters correctly" +f"Only {', '.join([f'<{p}>' for p in expected_initB])}")
                self.initB="unknown"
            

            self.B = hyperparameters['B']

        expected_hyperparameters = ('features', 'loss', 'dimension', 'source_dimension','B','save_b','kernelsettings','init_b','saveparam')
        unexpected_hyperparameters = set(hyperparameters.keys()).difference(expected_hyperparameters)
        if len(unexpected_hyperparameters) > 0:
            raise ValueError(f"Only {', '.join([f'<{p}>' for p in expected_hyperparameters])} are valid hyperparameters "
                             f"for an AbstractMultivariateModel! Unknown hyperparameters: {unexpected_hyperparameters}.")

    
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
        list_model_parameters_save=self.saveParam.copy()

        if with_mixing_matrix:
            model_parameters_save['mixing_matrix'] = self.attributes.mixing_matrix
            
        for i in range(len(list_model_parameters_save)):
            for key, value in list_model_parameters_save[i].items():
                if type(value) in [torch.Tensor]:
        
                    list_model_parameters_save[i][key] = value.tolist()

        
        
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()
       
        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            'features': self.features,
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'loss': self.loss,
            'parameters': model_parameters_save,
            'save_b':self.saveB, #faire une fonction pour recontruire B à partir de save B
            'init_b':self.initB,
            'kernelsettings':self.kernelsettings,
            'saveparam':list_model_parameters_save
        }
        print(model_settings)
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def compute_individual_tensorized_linear(self, timepoints, ind_parameters, attribute_type=None):
        
        # Population parameters
        positions, velocities, mixing_matrix = self._get_attributes(attribute_type)
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        velocities = velocities.reshape(1, 1, -1) # not needed in fact (automatic broadcasting on last dimension)
        positions = positions.reshape(1, 1, -1) # same
        reparametrized_time = reparametrized_time.unsqueeze(-1)

        # Computation
        LL = velocities * reparametrized_time + positions

        if self.source_dimension != 0:
            sources = ind_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())
            LL += wi.unsqueeze(-2)

        
        return LL # (n_individuals, n_timepoints, n_features)

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):

        return self.B(self.compute_individual_tensorized_linear(timepoints, ind_parameters, attribute_type=attribute_type))

    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        #à modifier
        # Population parameters
        positions, velocities, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)
        v0 = velocities.reshape(1, 1, -1) * torch.ones_like(reparametrized_time) # broadcast

        LL = v0 * reparametrized_time + positions
        if self.source_dimension != 0:
            sources = ind_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())
            LL += wi.unsqueeze(-2) # unsqueeze for (n_timepoints)

        alpha = torch.exp(xi).reshape(-1, 1, 1)

        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1), # add a last dimension for len param
            'tau': (-v0 * alpha).unsqueeze(-1), # same
        }

        if self.source_dimension > 0:
            derivatives['sources'] = mixing_matrix.expand((1,1,-1,-1)) * torch.ones_like(reparametrized_time).unsqueeze(-1) # broadcast on n_timepoints

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    

    

    """
    def compute_individual_tensorized_mixed(self, timepoints, ind_parameters, attribute_type=None):


        raise ValueError("Do not use !!!")

        # Hyperparameters : split # TODO
        split = 1
        idx_linear = list(range(split))
        idx_logistic = list(range(split, self.dimension))

        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau, sources = ind_parameters['xi'], ind_parameters['tau'], ind_parameters['sources']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.reshape(*timepoints.shape, 1)
        v0 = v0.reshape(1, 1, -1)

        LL = v0 * reparametrized_time
        if self.source_dimension != 0:
            wi = sources.matmul(a_matrix.t())
            LL += wi.unsqueeze(-2)

        # Logistic Part
        LL_log = 1. + g * torch.exp(-LL * b)
        model_logistic = (1. / LL_log)[:,:,idx_logistic]

        # Linear Part
        model_linear = (LL + torch.log(g))[:,:,idx_linear]

        # Concat
        model = torch.cat([model_linear, model_logistic], dim=2)

        return model
    """

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self, set_v0_prior = False):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01}, # population parameters
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # Initialize hyperpriors
        if set_v0_prior:
            self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone().detach()
            self.MCMC_toolbox['priors']['s_v0'] = 0.1
            # same on g?

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('v0', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['v0'].tensor_realizations = realizations['v0'].tensor_realizations + mean_xi

        self.update_MCMC_toolbox(['v0'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations):

        # <!> by doing this here, we change v0 and thus orthonormal basis and mixing matrix,
        #     the betas / sources are not related to the previous orthonormal basis...
        realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations,
            'v0': realizations['v0'].tensor_realizations,
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

        norm_1 = data.values * data_reconstruction
        norm_2 = data_reconstruction * data_reconstruction

        sufficient_statistics['obs_x_reconstruction'] = norm_1  # .sum(dim=2) # no sum on features...
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2  # .sum(dim=2) # no sum on features...

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters,
                                                                                                  attribute_type="MCMC")

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):

        # <!> by doing this here, we change v0 and thus orthonormal basis and mixing matrix,
        #     the betas / sources are not related to the previous orthonormal basis...
        realizations = self._center_xi_realizations(realizations)

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations.detach()

        v0_emp = realizations['v0'].tensor_realizations.detach()
        if self.MCMC_toolbox['priors'].get('v0_mean', None) is not None:
            v0_mean = self.MCMC_toolbox['priors']['v0_mean']
            s_v0 = self.MCMC_toolbox['priors']['s_v0']
            sigma_v0 = self.MCMC_toolbox['priors']['v0_std']
            self.parameters['v0'] = (1 / (1 / (s_v0 ** 2) + 1 / (sigma_v0 ** 2))) * (
                        v0_emp / (sigma_v0 ** 2) + v0_mean / (s_v0 ** 2))
        else:
            # new default
            self.parameters['v0'] = v0_emp

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        # self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
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
        self.parameters['v0'] = suff_stats['v0']
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

        v0_infos = {
            "name": "v0",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
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
            "shape": torch.Size([self.source_dimension]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "v0": v0_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        return variables_infos
