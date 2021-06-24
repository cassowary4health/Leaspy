import torch

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_factory import AttributesFactory

from .utils.initialization.model_initialization import initialize_parameters
from leaspy import __version__
import json

class LogisticAsymp(AbstractMultivariateModel):
    """
    Logistic model for multiple variables of interest.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.neg=False
        self.max_asymp=1.1
        self.parameters["Param"] = None
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
            'neg':self.neg,
            'max_asymp':self.max_asymp,
            'parameters': model_parameters_save,
            'save_b':self.saveB, 
            'init_b':self.initB,
            'kernelsettings':self.kernelsettings,
            'saveparam':list_model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,self.neg,self.max_asymp)
        self.attributes.update(['all'], self.parameters)

    def initialize(self, dataset, method="default"):
        self.dimension = dataset.dimension
        self.features = dataset.headers

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))

        self.parameters = initialize_parameters(self, dataset, method)
        

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,self.neg,self.max_asymp)
        self.attributes.update(['all'], self.parameters)
        self.is_initialized = True

   

    def compute_individual_tensorized_preB(self, timepoints, ind_parameters, attribute_type=None):
        # Population parameters
        g, Param, a_matrix = self._get_attributes(attribute_type)
        
        
        #slicer Param
        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        Infection=Param[:self.dimension]
        Rho=Param[self.dimension:]
        a=self.max_asymp
        Rho[Rho!=Rho]=a
        Rho=torch.clamp(Rho,max=a-0.001)
       
        
        #Rho[Rho<0]=0.
        
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)
        
        
        LL = Infection * reparametrized_time
       
        sources = ind_parameters['sources']
       
        if self.source_dimension != 0 and len(sources.shape)>1:
            
            wi = sources.matmul(a_matrix.t())
            infection_w=wi[:,:self.dimension]

            #deltaw=infection_w-guerison_w
            quo=a-Rho
            Rhow=(a+1/quo)/Rho

            Rho1=Rhow/(1+(Rhow/Rho-1)*torch.exp(-infection_w))
            p1=Rhow/(1+(Rhow*(1+g)/Rho-1)*torch.exp(-infection_w))
            g1=Rho1/p1-1
           
           
            LL=LL.permute(1,0,2)
                
            LL = 1.0 + g1 * torch.exp(-LL)
            model = Rho1 / LL
            model=model.permute(1,0,2)
                
            return model
           

        else:
            LL = 1.0 + g * torch.exp(-LL)
            model = Rho / LL
            
            return model

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
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,self.neg,self.max_asymp)
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
            "shape": torch.Size([2*self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension, self.source_dimension]),
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
            "Param": Param_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos
        return variables_infos
