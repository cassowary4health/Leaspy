from abc import abstractmethod
import json
import math

import torch

from leaspy import __version__

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.utils.docs import doc_with_super

import leaspy.models.utils.OptimB as OptimB
initB = {"identity":torch.nn.Identity(),
"negidentity":lambda x:-x,
"logistic":lambda x:1./(1.+torch.exp(-x))}

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
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']

        expected_hyperparameters = ('features', 'loss', 'dimension', 'source_dimension')
        unexpected_hyperparameters = set(hyperparameters.keys()).difference(expected_hyperparameters)
        if len(unexpected_hyperparameters) > 0:
            raise ValueError(
                f"Only {', '.join([f'<{p}>' for p in expected_hyperparameters])} are valid hyperparameters "
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

        if self.name=="logistic_asymptots_delay":
            individual_parameters["sources_asymptots"]=torch.zeros(self.source_dimension, dtype=torch.float32)

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def _get_attributes(self, attribute_type):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise ValueError("The specified attribute type does not exist : {}".format(attribute_type))
