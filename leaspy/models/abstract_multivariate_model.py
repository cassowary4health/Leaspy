from abc import abstractmethod
import json
import math

import torch

from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.utils.docs import doc_with_super

initB = {"identity":torch.nn.Identity(),
"negidentity":lambda x:-x,
"logistic":lambda x:1./(1.+torch.exp(-x))}
import leaspy.models.utils.OptimB as OptimB

@doc_with_super()
class AbstractMultivariateModel(AbstractModel):
    """
    Contains the common attributes & methods of the multivariate models.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.source_dimension = None
        self.dimension = None
        self.B= lambda x:x
        self.moving_b=None
        self.kernelsettings=None
        self.saveParam=[]
        self.saveB= []
        self.initB= "identity"
        
        self.parameters = {
            "g": None,
            "betas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "sources_mean": None, "sources_std": None,
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
                'betas_std': None
            }
        }

        # load hyperparameters
        self.load_hyperparameters(kwargs)

    """
    def smart_initialization_realizations(self, data, realizations):
        # TODO : Qui a fait ça? A quoi ça sert?
        # means_time = torch.tensor([torch.mean(data.get_times_patient(i)) for
        # i in range(data.n_individuals)]).reshape(realizations['tau'].tensor_realizations.shape)
        # realizations['tau'].tensor_realizations = means_time
        return realizations
    """
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

    def initialize(self, dataset, method="default"):
        self.dimension = dataset.dimension
        if self.moving_b is None:
            self.moving_b=[i for i in range(self.dimension)]
        self.features = dataset.headers

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))

        self.parameters = initialize_parameters(self, dataset, method)
        

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)
        self.is_initialized = True

    @abstractmethod
    def initialize_MCMC_toolbox(self):
        """
        Initialize Monte-Carlo Markov-Chain toolbox for calibration of model

        TODO to move in a "MCMC-model interface"
        """
        pass

    @abstractmethod
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        """
        Update the MCMC toolbox with a collection of realizations of model population parameters.

        TODO to move in a "MCMC-model interface"

        Parameters
        ----------
        name_of_the_variables_that_have_been_changed: container[str] (list, tuple, ...)
            Names of the population parameters to update in MCMC toolbox
        realizations : :class:`.CollectionRealization`
            All the realizations to update MCMC toolbox with
        """
        pass

    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'init' in hyperparameters.keys():
            self.init = hyperparameters['init']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']
        if 'neg' in hyperparameters.keys():
            self.neg = hyperparameters['neg']
        if 'max_asymp' in hyperparameters.keys():
            self.max_asymp = hyperparameters['max_asymp']
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
            self.saveParam=hyperparameters['moving_b']
            
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

        expected_hyperparameters = ('features', 'loss', 'dimension', 'source_dimension','neg','max_asymp','init','source_dimension_direction','B','save_b','kernelsettings','init_b','saveparam','moving_B')
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
            'save_b':self.saveB, 
            'init_b':self.initB,
            'kernelsettings':self.kernelsettings,
            'saveparam':list_model_parameters_save,
            'moving_b':self.moving_b
        }
       
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)


    
    def compute_individual_tensorized_preb(self, timepoints, individual_parameters, attribute_type=None):
        raise NotImplementedError
    
    
    def compute_individual_tensorized(self, timepoints, individual_parameters, attribute_type=None):
            A=self.compute_individual_tensorized_preb(timepoints, individual_parameters, attribute_type=attribute_type)
            B=self.B(A)
            if len(B.shape)==3:
                A[:,:,self.moving_b]=B[:,:,self.moving_b]
            elif len(B.shape)==2:
                A[:,self.moving_b]=B[:,self.moving_b]
            else :
                A[self.moving_b]=B[:,self.moving_b]

            return A

    def compute_mean_traj(self, timepoints):
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` [1, n_timepoints]

        Returns
        -------
        :class:`torch.Tensor` [1, n_timepoints, dimension]
            The group-average values at given timepoints
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32),
            'tau': torch.tensor([self.parameters['tau_mean']], dtype=torch.float32),
            'sources': torch.zeros(self.source_dimension, dtype=torch.float32)
        }
        if self.name=="logistic_asymp_delay":
            individual_parameters["sources_asymp"]=torch.zeros(self.source_dimension_asymp, dtype=torch.float32)

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def _get_attributes(self, attribute_type):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise ValueError("The specified attribute type does not exist : {}".format(attribute_type))
