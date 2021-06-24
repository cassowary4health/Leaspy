import torch
import json

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_factory import AttributesFactory
from .utils.initialization.model_initialization import initialize_parameters
from leaspy import __version__

expected_initB=["identity","negidentity","logistic"]
initB = {"identity":torch.nn.Identity(),
"negidentity":lambda x:-x,
"logistic":lambda x:1./(1.+torch.exp(-x))}
class LogisticAsympDelay(AbstractMultivariateModel):
    """
    Logistic asympdelay model for multiple variables of interest.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.neg=False
        self.parameters["Param"] = None
        self.MCMC_toolbox['priors']['Param_std'] = None  # Value, Coef
        self.max_asymp=1.1
        if self.dimension is not None:
            self.features_moving_asymp=[i for i in range(self.dimension)]#list of indexes
        else:
            self.features_moving_asymp=None
        self.source_dimension_asymp=self.source_dimension
        self.B= torch.nn.Identity()

    def initialize(self, dataset, method="default"):
        self.dimension = dataset.dimension
        if self.features_moving_asymp is None:
            self.features_moving_asymp=[i for i in range(self.dimension)]
            
        self.features = dataset.headers

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))

        self.parameters = initialize_parameters(self, dataset, method)
        

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)
        self.is_initialized = True

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
            'moving_b':self.moving_b,
            'kernelsettings':self.kernelsettings,
            'saveparam':list_model_parameters_save,
            'source_dimension_asymp':self.source_dimension_asymp,
            'features_moving_asymp':self.features_moving_asymp
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)
    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'init' in hyperparameters.keys():
            self.init = hyperparameters['init']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']
            if self.source_dimension_asymp is None:
                self.source_dimension_asymp =self.source_dimension 
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']
        if 'neg' in hyperparameters.keys():
            self.neg = hyperparameters['neg']
        if 'max_asymp' in hyperparameters.keys():
            self.max_asymp = hyperparameters['max_asymp']
        if 'source_dimension_asymp'in hyperparameters.keys():
            self.source_dimension_asymp = hyperparameters['source_dimension_asymp']
        if 'features_moving_asymp'in hyperparameters.keys():
            if len(hyperparameters['features_moving_asymp'])==0:
                self.features_moving_asymp= []
            elif type(hyperparameters['features_moving_asymp'][0]) is str:
                L=[]
                for i,e in enumerate(self.features):
                    if e in hyperparameters['features_moving_asymp']:
                        L.append(i)
                self.features_moving_asymp = L
            elif type(hyperparameters['features_moving_asymp'][0]) is int:
                self.features_moving_asymp = hyperparameters['features_moving_asymp']
            else:
                raise ValueError("you should give the list of features you want to have their asymptots not fixed, or the list of their indexes")
        if 'source_dimension_direction' in hyperparameters.keys():
            self.source_dimension_direction = hyperparameters['source_dimension_direction']
        if 'kernelsettings' in hyperparameters.keys():
            self.kernelsettings = hyperparameters['kernelsettings']
        if 'init_b' in hyperparameters.keys():
            self.initB=hyperparameters['init_b']
            self.initBlink()
        if 'saveparam' in hyperparameters.keys():
            self.saveParam=hyperparameters['saveparam']
        if 'moving_b' in hyperparameters.keys():
            self.moving_b=hyperparameters['moving_b']
            
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

        expected_hyperparameters = ('features', 'loss', 'dimension', 'source_dimension','neg','max_asymp','init','source_dimension_direction','B','moving_b','save_b','kernelsettings','init_b','saveparam','features_moving_asymp','source_dimension_asymp')
        unexpected_hyperparameters = set(hyperparameters.keys()).difference(expected_hyperparameters)
        if len(unexpected_hyperparameters) > 0:
            raise ValueError(f"Only {', '.join([f'<{p}>' for p in expected_hyperparameters])} are valid hyperparameters "
                             f"for an AbstractMultivariateModel! Unknown hyperparameters: {unexpected_hyperparameters}.")


    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension,self.neg,self.max_asymp)
        self.attributes.update(['all'], self.parameters)

   

    def compute_individual_tensorized_preb(self, timepoints, ind_parameters, attribute_type=None):
        # Population parameters
        g, Param,Asymp, a_matrix_delay,a_matrix_asymp = self._get_attributes(attribute_type)
        
        
        #slicer Param
        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        Infection=Param
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)
        
        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features)
        
        
        LL = Infection * reparametrized_time
       
        sources_delay = ind_parameters['sources']

        sources_asymp = ind_parameters['sources_asymp']
        Rho=torch.ones(self.dimension)
        Rho[self.features_moving_asymp]=Asymp
        a=self.max_asymp
        if (Rho>a).any():
            LL = 1.0 + g * torch.exp(-LL)
            model = Rho / LL
            
            return model
       
        if self.source_dimension != 0 and len(sources_delay.shape)>1:
            
            wi_delay = sources_delay.matmul(a_matrix_delay.t())
            wi_asymp = sources_asymp.matmul(a_matrix_asymp.t())
            
            infection_w=wi_asymp
            LL+=wi_delay.unsqueeze(-2)
          
            quo=a-Asymp
            Rhow=(a+1/quo)/Asymp
            gs=g[self.features_moving_asymp]

            Rho1=Rhow/(1+(Rhow/Asymp-1)*torch.exp(-infection_w))
            p1=Rhow/(1+(Rhow*(1+gs)/Asymp-1)*torch.exp(-infection_w))
            g1=Rho1/p1-1

            LL=LL.permute(1,0,2)
            
            LL1 = 1.0 + g1 * torch.exp(-LL[:,:,self.features_moving_asymp])
            model_moving = Rho1 / LL1
            LL2 = 1.0 + g * torch.exp(-LL)
            model= Rho / LL2
            model[:,:,self.features_moving_asymp]=model_moving
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
            'priors': {'g_std': 0.01, 'Param_std': 0.01, 'betas_std': 0.01,'Asymp_std':0.01,'betas_asymp_std':0.01},
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # TODO maybe not here
        # Initialize priors
        self.MCMC_toolbox['priors']['Param_mean'] = self.parameters['Param'].clone()
        self.MCMC_toolbox['priors']['s_Param'] = 0.1
        self.MCMC_toolbox['priors']['Asymp_mean'] = self.parameters['Asymp'].clone()
        self.MCMC_toolbox['priors']['s_Asymp'] = 0.1

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('Param', 'all')):
            values['Param'] = realizations['Param'].tensor_realizations
        if any(c in L for c in ('Asymp', 'all')):
            values['Asymp'] = realizations['Asymp'].tensor_realizations
        if any(c in L for c in ('betas_asymp', 'all')):
            values['betas_asymp'] = realizations['betas_asymp'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        #On recentre les ci de cette manière en vue de l'argument de l'exponentielle au dénominateur
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['Param'].tensor_realizations= realizations['Param'].tensor_realizations + mean_xi
        #On modifie les self.dimension premiers termes car c'est les taux d'infection
        self.update_MCMC_toolbox(['all'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations,
            'Param': realizations['Param'].tensor_realizations,
            'Asymp': realizations['Asymp'].tensor_realizations,
            'tau': realizations['tau'].tensor_realizations,
            'tau_sqrd': torch.pow(realizations['tau'].tensor_realizations, 2),
            'xi': realizations['xi'].tensor_realizations,
            'xi_sqrd': torch.pow(realizations['xi'].tensor_realizations, 2)
        }
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations
        if self.source_dimension_asymp != 0:
            sufficient_statistics['betas_asymp'] = realizations['betas_asymp'].tensor_realizations

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
        if self.MCMC_toolbox['priors']['Asymp_mean'] is not None:
            Asymp_mean = self.MCMC_toolbox['priors']['Asymp_mean']
            Asymp_emp = realizations['Asymp'].tensor_realizations
            s_Asymp = self.MCMC_toolbox['priors']['s_Asymp']
            sigma_Asymp = self.MCMC_toolbox['priors']['Asymp_std']
            self.parameters['Asymp'] = (1 / (1 / (s_Asymp ** 2) + 1 / (sigma_Asymp** 2))) * (
                        Asymp_emp / (sigma_Asymp ** 2) + Asymp_mean / (s_Asymp ** 2))
        else:
            self.parameters['Asymp'] = realizations['Asymp'].tensor_realizations

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations
        if self.source_dimension_asymp != 0:
            self.parameters['betas_asymp'] = realizations['betas_asymp'].tensor_realizations
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
        self.parameters['Asymp'] = suff_stats['Asymp']
        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']
        if self.source_dimension_asymp != 0:
            self.parameters['betas_asymp'] = suff_stats['betas_asymp']

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
        Asymp_infos = {
            "name": "Asymp",
            "shape": torch.Size([len(self.features_moving_asymp)]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension-1, self.source_dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_asymp_infos = {
            "name": "betas_asymp",
            "shape": torch.Size([len(self.features_moving_asymp), self.source_dimension_asymp]),
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
        sources_asymp_infos = {
            "name": "sources_asymp",
            "shape": torch.Size([self.source_dimension_asymp]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "Param": Param_infos,
            "Asymp":Asymp_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        if self.source_dimension_asymp != 0:
            variables_infos['sources_asymp'] = sources_asymp_infos
            variables_infos['betas_asymp'] = betas_asymp_infos
        return variables_infos
