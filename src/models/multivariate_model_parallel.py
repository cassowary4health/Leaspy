import numpy as np
import torch
from src.models.utils.attributes.attributes_multivariateparallel import Attributes_MultivariateParallel

from src.models.abstract_model import AbstractModel

class MultivariateModelParallel(AbstractModel):
    # TODO : Remove call to the Abstract Model
    def __init__(self):
        self.model_name = 'Multivariate_Parallel'
        self.dimension = None
        self.source_dimension = None
        self.is_initialized = False
        self.parameters = {
            "g": None, "betas": None, "deltas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None,  "xi_std": None,
            "sources_mean": None, "sources_std": None,
            "noise_std": None

        }
        self.bayesian_priors = None
        self.attributes = None

        ### MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'g_std': None, # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
                'deltas_std': None, # tq deltas = user_delta_in_years * v0 / (p0(1-p0))
                'betas_std': None
            }
        }



    def load_parameters(self, parameters):
        for k in self.parameters.keys():
            if k != 'p0':
                # TODO Check that everything is in it
                self.parameters[k] = parameters[k]
            else:
                self.parameters['g'] = np.log(1/parameters['p0'] - 1)

    def save_parameters(self, parameters):
        #TODO
        return 0


    def initialize_parameters(self, dataset, smart_initialization):
        # TODO : have the else/if bifurcation in the mother class
        # TODO : Load the values of the parameters from a default file
        # TODO : The initialize parameters should not stay here
        if smart_initialization:
            if self.is_initialized:
                print("Your parameters were already initialized - they have been overwriten")
            #TODO TODO : Smart initialization
            self.dimension = dataset.dimension
            self.source_dimension = self.dimension - 1
            self.parameters = {'g': 0.5, 'tau_mean': 70.0, 'tau_std': 2.0, 'xi_mean': -3., 'xi_std': 0.1,
                               'sources_mean': 0.0,
                               'sources_std': 1.0,
                               'noise_std': 0.1, 'deltas': [-3., -2., -2.],
                               'betas': np.zeros((self.dimension - 1, self.source_dimension)).tolist()
                               }
        else:
            if self.is_initialized:
                pass
            else:
                self.dimension = dataset.dimension
                self.source_dimension = self.dimension - 1
                self.parameters = {'g': 0.5, 'tau_mean': 70.0, 'tau_std': 2.0, 'xi_mean': -3., 'xi_std': 0.1,
                               'sources_mean': 0.0,
                               'sources_std': 1.0,
                               'noise_std': 0.1, 'deltas': [-3., -2., -2.],
                               'betas': np.zeros((self.dimension - 1, self.source_dimension)).tolist()
                               }

        # TODO self.attributes.update(self.parameters)name_of_changed_realizations

        # TODO : check that the parameters have really been initialized and that they are all here!
        self.is_initialized = True
        self.attributes = Attributes_MultivariateParallel(self.dimension, self.source_dimension)

    def initialize_MCMC_toolbox(self, dataset):

        self.MCMC_toolbox['priors'] = {
            'g_std': 0.01,
            'deltas_std': 0.01,
            'betas_std': 0.01
        }

        self.MCMC_toolbox['attributes'] = Attributes_MultivariateParallel(self.dimension, self.source_dimension)

        values = {
            'g': self.parameters['g'],
            'deltas': self.parameters['deltas'],
            'betas': self.parameters['betas']
        }
        values['tau_mean'] = self.parameters['tau_mean']
        values['xi_mean'] = self.parameters['xi_mean']
        self.MCMC_toolbox['attributes'].update(['all'], values)


    def update_MCMC_toolbox(self, name_of_the_variable_that_has_been_changed, realizations):
        """
        :param new_realizations: {('name', position) : new_scalar_value}
        :return:
        """
        ### TODO : Check if it is possible / usefull to have multiple variables sampled

        # Updates the attributes of the MCMC_toolbox

        ### TODO : Probably convert all the variables to torch tensors
        values = {
            'g': realizations['g'].tensor_realizations.detach().numpy(),
            'deltas': realizations['deltas'].tensor_realizations.detach().numpy(),
            'xi_mean': self.parameters['xi_mean'],
            'betas': realizations['betas'].tensor_realizations.detach().numpy()
        }

        self.MCMC_toolbox['attributes'].update([name_of_the_variable_that_has_been_changed],
                                               values)



    def compute_loglikelihood_MCMC(self, individual_parameters):
        ### TODO Compute the likelihood during the MCMC phase
        ### TODO It uses the realization of the random variables
        return 0


    def compute_parallel_curve(self, age, xi, tau, sources, attributes=False):
        ### TODO IMPORTANT : There is a need - for the other important function such as leaspy.predict or leaspy.simulate
        ### TODO : to have a fit(age, xi, tau, sources) function that internally uses the self.attributes parameters
        ### TODO : as this need to be a UNIQUE function to do that everywhere
        ### TODO HINT : there is probably a need for all the "attributes" that are in the realizations, parameters, RV, ...
        ### TODO HINT : to be stored in the attributes object. This way, the function fit can have a default attribute value
        ### TODO HINT : at False that uses the internal attibutes. Otherwise, it uses the attributes that are passed as
        ### TODO HINT : parameters of the fit function.
        ### TODO : This function should be vectorized so that it can be computed on a single individual or multiple individuals
        if attributes == False:
            attributes = self.attributes

        g = attributes['g']
        deltas = attributes['deltas']
        mixing_matrix = attributes['mixing_matrix']

        _unique_indiv = True
        if _unique_indiv:
            reparametrized_time = np.exp(xi)* (age - tau)
            wi = np.dot(mixing_matrix, sources)
            eta = - (g * np.exp(-deltas) + 1)**2 / (g*np.exp(-deltas))
            eta = -eta * wi - deltas - reparametrized_time
            eta = 1./(1. + g * np.exp(eta))
        else:
            ### TODO : Here goes the tensorized version
            pass

        return eta



    def compute_individual_tensorized(self, data, realizations):
        # TODO 1 : Check later if the usage of the attributes of the model improves the speed of the algorithm
        # TODO 1 : by being precalculated
        # TODO 2 : Maybe used nowhere except in the compute_individual_attachment
        g = torch.exp(realizations['g'].tensor_realizations)
        timepoints = data.timepoints.reshape(data.timepoints.shape[0], data.timepoints.shape[1], 1)
        deltas = torch.cat([torch.Tensor([0.0]).reshape(1, 1), realizations["deltas"].tensor_realizations], dim=1)
        a_matrix = self.MCMC_toolbox['attributes'].mixing_matrix
        wi = torch.nn.functional.linear(realizations['sources'].tensor_realizations, a_matrix, bias=None)
        #print(realizations['xi']._tensor_realizations.shape, realizations['tau'].shape, timepoints.shape)
        reparametrized_time = torch.exp(realizations['xi'].tensor_realizations) * (
                    timepoints - realizations['tau'].tensor_realizations)
        deltas_exp = torch.exp(-deltas)

        b = wi * (g * deltas_exp + 1) ** 2 / (g * deltas_exp)

        a = -reparametrized_time - deltas - b
        a = 1 + g * torch.exp(a)
        model = 1 / a

        return model * data.mask

    def compute_individual_attachment_tensorized(self, data, realizations):
        data_fit = self.compute_individual_tensorized(data, realizations)
        sum_squared = ((data_fit - data.values) ** 2).sum(dim=(1, 2))
        attachment = 0.5 * (1/self.parameters['noise_std']**2) * sum_squared
        attachment += np.log(np.sqrt(2 * np.pi * self.parameters['noise_std']**2))

        return attachment





    def random_variable_informations(self):
        # TODO : Change the name of this method

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": (1, 1),
            "type": "population",
            "rv_type": "multigaussian"
        }

        deltas_infos = {
            "name": "deltas",
            "shape": (1, self.dimension-1),
            "type": "population",
            "rv_type": "multigaussian"
        }


        betas_infos = {
            "name": "betas",
            "shape": (self.dimension - 1, self.source_dimension),
            "type": "population",
            "rv_type": "multigaussian"
        }

        ## Individual variables
        tau_infos = {
            "name": "tau",
            "shape": (1, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        xi_infos = {
            "name": "xi",
            "shape": (1, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        sources_infos = {
            "name": "sources",
            "shape": (1, self.source_dimension),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "deltas": deltas_infos,
            "betas": betas_infos,
            "tau": tau_infos,
            "xi": xi_infos,
            "sources": sources_infos
        }

        return variables_infos



    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations.detach().numpy()
        sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations.detach().numpy()
        sufficient_statistics['betas'] = realizations['betas'].tensor_realizations.detach().numpy()
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi_sqrd'].tensor_realizations, 2)

        ## TODO : To finish
        data_reconstruction = self.compute_individual_tensorized(data, realizations)
        data_real = data.values
        norm_1 = data_real * data_reconstruction
        norm_2 = data_reconstruction * data_reconstruction
        sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=2)
        sufficient_statistics['reconstruction_sqrd'] = torch.sum(norm_2, dim=2)

        return sufficient_statistics

    def update_model_parameters(self, data, sufficient_statistics, burn_in_phase=True):

        ## Probably optimize some common parts

        # Memoryless part of the algorithm
        if burn_in_phase:
            self.parameters['g'] = sufficient_statistics['g'].tensor_realizations.detach().numpy()
            self.parameters['deltas'] = sufficient_statistics['deltas'].tensor_realizations.detach().numpy()
            self.parameters['betas'] = sufficient_statistics['betas'].tensor_realizations.detach().numpy()
            xi = sufficient_statistics['xi'].tensor_realizations.detach().numpy()
            self.parameters['xi_mean'] = np.mean(xi)
            self.parameters['xi_std'] = np.std(xi)
            tau = sufficient_statistics['tau'].tensor_realizations.detach().numpy()
            self.parameters['tau_mean'] = np.mean(tau)
            self.parameters['tau_std'] = np.std(tau)

            data_fit = self.compute_individual_tensorized(data, sufficient_statistics)
            #norm_of_tensor = torch.norm(data.values - data_fit, p=2, dim=2).sum()
            squared_diff = ((data_fit-data.values)**2).sum()
            self.parameters['noise_std'] = np.sqrt(squared_diff/(data.n_visits*data.dimension))

        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            ## TODO : To finish
            self.parameters['g'] = sufficient_statistics['g']
            self.parameters['deltas'] = sufficient_statistics['deltas']
            self.parameters['betas']  = sufficient_statistics['betas']
            self.parameters['xi_mean'] = np.mean(sufficient_statistics['xi'])
            self.parameters['xi_std'] = np.sqrt(np.mean(sufficient_statistics['xi_sqrd']) - np.sum(sufficient_statistics['xi'])**2)
            self.parameters['tau_mean'] = np.mean(sufficient_statistics['tau'])
            self.parameters['tau_std'] = np.sqrt(np.mean(sufficient_statistics['tau_sqrd']) - np.sum(sufficient_statistics['tau'])**2)
            self.parameters['noise_std'] = 0.01



