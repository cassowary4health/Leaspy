import json

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import torch

from .abstract_model import AbstractModel
from .utils.attributes.attributes_factory import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters


class UnivariateModel(AbstractModel):
    """
    Univariate (logistic or linear) model for a single variable of interest.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dimension = 1
        self.source_dimension = 0  # TODO, None ???
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

        # load hyperparameters
        self.load_hyperparameters(kwargs)

    def save(self, path, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path: str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for json.dump method.
        """
        model_parameters_save = self.parameters.copy()
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'name': self.name,
            'features': self.features,
            #'dimension': 1,
            'loss': self.loss,
            'parameters': model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def load_hyperparameters(self, hyperparameters):
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        if 'loss' in hyperparameters.keys():
            self.loss = hyperparameters['loss']
        if any([key not in ('features', 'loss') for key in hyperparameters.keys()]):
            raise ValueError("Only <features> and <loss> are valid hyperparameters for an UnivariateModel!"
                             f"You gave {hyperparameters}.")

    def initialize(self, dataset, method="default"):

        # "Smart" initialization : may be improved
        # TODO !
        self.features = dataset.headers

        """
        self.parameters = {
            'g': [1.],
            'tau_mean': 70., 'tau_std': 2.,
            'xi_mean': -3., 'xi_std': .1,
            'noise_std': [.1]}
        self.parameters = {key: torch.tensor(val) for key, val in self.parameters.items()}
        self.attributes = AttributesFactory.attributes(self.name, dimension=1)
        """

        self.parameters = initialize_parameters(self, dataset, method)

        self.attributes = AttributesFactory.attributes(self.name, dimension=1)
        self.attributes.update(['all'], self.parameters)

        self.is_initialized = True

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, dimension=1)
        self.attributes.update(['all'], self.parameters)

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 1.},
            'attributes': AttributesFactory.attributes(self.name, dimension=1)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    ##########
    # CORE
    ##########
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('xi_mean', 'all')):
            values['xi_mean'] = self.parameters['xi_mean']

        self.MCMC_toolbox['attributes'].update(L, values)

    def _get_attributes(self, MCMC):
        if MCMC:
            g = self.MCMC_toolbox['attributes'].positions
        else:
            g = self.attributes.positions
        return g

    # def compute_sum_squared_tensorized(self, data, param_ind, attribute_type):
    #    res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
    #    res *= data.mask
    #    return torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))

    # TODO generalize in abstract
    def compute_mean_traj(self, timepoints):
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32),
            'tau': torch.tensor([self.parameters['tau_mean']], dtype=torch.float32),
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def plot_param_ind(self, path, param_ind):
        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        fig, ax = plt.subplots(1, 1)
        xi, tau = param_ind
        ax.plot(xi.squeeze(1).detach().tolist(), tau.squeeze(1).detach().tolist(), 'x')
        plt.xlabel('xi')
        plt.ylabel('tau')
        pdf.savefig(fig)
        plt.close()
        pdf.close()

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name == 'univariate_logistic':
            return self.compute_individual_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name == 'univariate_linear':
            return self.compute_individual_tensorized_linear(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError("Mutivariate model > Compute individual tensorized")

    def compute_individual_tensorized_logistic(self, timepoints, ind_parameters, attribute_type=False):
        # Population parameters
        g = self._get_attributes(attribute_type)
        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        LL = -reparametrized_time.unsqueeze(-1)
        model = 1. / (1. + g * torch.exp(LL))

        return model

    def compute_individual_tensorized_linear(self, timepoints, ind_parameters, attribute_type=False):
        # Population parameters
        positions = self._get_attributes(attribute_type)
        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)
        LL = -reparametrized_time.unsqueeze(-1)
        model = positions - LL

        return model

    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name in ['logistic', 'univariate_logistic']:
            return self.compute_jacobian_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name in ['linear', 'univariate_linear']:
            return self.compute_jacobian_tensorized_linear(timepoints, ind_parameters, attribute_type)
        elif self.name == 'mixed_linear-logistic':
            return self.compute_jacobian_tensorized_mixed(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError("Mutivariate model > Compute jacobian tensorized")

    def compute_jacobian_tensorized_linear(self, timepoints, ind_parameters, attribute_type=None):
        '''
        Parameters
        ----------
        timepoints
        ind_parameters
        attribute_type

        Returns
        -------
        The Jacobian of the model with parameters order : [xi, tau, sources].
        This function aims to be used in scipy_minimize.
        '''
        # Population parameters
        positions = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1)

        LL = reparametrized_time + positions

        alpha = torch.exp(xi).unsqueeze(-1)

        derivatives = {
            'xi' : (reparametrized_time).unsqueeze(-1),
            'tau' : (-alpha * torch.ones_like(reparametrized_time)).unsqueeze(-1),
        }
        return derivatives

    def compute_jacobian_tensorized_logistic(self, timepoints, ind_parameters, MCMC=False):
        # cf. AbstractModel.compute_jacobian_tensorized for doc

        # Population parameters
        g = self._get_attributes(MCMC)

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.unsqueeze(-1) # (n_individuals, n_timepoints, n_features==1)

        model = 1. / (1. + g * torch.exp(-reparametrized_time))

        c = model * (1. - model)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        derivatives = {
            'xi': (c * reparametrized_time).unsqueeze(-1),
            'tau': (c * -alpha).unsqueeze(-1),
        }

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations.detach() # avoid 0D / 1D tensors mix
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        ind_parameters = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, ind_parameters, attribute_type=True)

        data_reconstruction *= data.mask.float() # speed-up computations
        #norm_0 = data.values * data.values * data.mask.float()
        norm_1 = data.values * data_reconstruction #* data.mask.float()
        norm_2 = data_reconstruction * data_reconstruction #* data.mask.float()
        #sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=2)
        sufficient_statistics['obs_x_reconstruction'] = norm_1 #.sum(dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = norm_2 #.sum(dim=2)

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters, attribute_type=True)

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type=True).sum()
        self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)

        if self.loss == 'crossentropy':
            crossentropy = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type=True).sum()
            self.parameters['crossentropy'] = crossentropy
        # Stochastic sufficient statistics used to update the parameters of the model

    def update_model_parameters_normal(self, data, suff_stats):
        self.parameters['g'] = suff_stats['g']

        tau_mean = self.parameters['tau_mean']
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(tau_std_updt + self.parameters['tau_mean'] ** 2)
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
        self.parameters['xi_std'] = torch.sqrt(xi_std_updt + self.parameters['xi_mean'] ** 2)
        self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        #S1 = torch.sum(suff_stats['obs_x_obs'])
        S1 = data.L2_norm
        S2 = suff_stats['obs_x_reconstruction'].sum()
        S3 = suff_stats['reconstruction_x_reconstruction'].sum()

        self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations)

        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()

    # def get_param_from_real(self,realizations):
    #    xi = realizations['xi'].tensor_realizations
    #    tau = realizations['tau'].tensor_realizations
    #    return (xi,tau)

    def param_ind_from_dict(self, individual_parameters):
        xi, tau = [], []
        for key, item in individual_parameters.items():
            xi.append(item['xi'])
            tau.append(item['tau'])
        xi = torch.tensor(xi, dtype=torch.float32).unsqueeze(1)
        tau = torch.tensor(tau, dtype=torch.float32).unsqueeze(1)
        return (xi, tau)

    def get_xi_tau(self, param_ind):
        xi, tau = param_ind
        return xi, tau

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

        return variables_infos
