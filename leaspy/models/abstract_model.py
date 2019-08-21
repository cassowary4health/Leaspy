import torch
from leaspy.utils.realizations.collection_realization import CollectionRealization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

class AbstractModel():
    def __init__(self, name):
        self.name = name
        self.parameters = None
        self.is_initialized = False

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = parameters[k]

    def load_hyperparameters(self, hyperparameters):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def initialize(self, dataset, is_initialized):
        raise NotImplementedError

    def get_xi_tau(self,param_ind):
        xi,tau,sources = param_ind
        return xi,tau

    def plot_param_ind(self,path,param_ind):
        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        fig, ax = plt.subplots(1, 1)
        xi,tau,sources = param_ind
        ax.plot(xi.squeeze(1).detach().numpy(),tau.squeeze(1).detach().numpy(),'x')
        plt.xlabel('xi')
        plt.ylabel('tau')
        pdf.savefig(fig)
        plt.close()
        for i in range(self.source_dimension):
            fig, ax = plt.subplots(1, 1)
            ax.plot(sources[:,i].detach().numpy(),'x')
            plt.title("sources "+str(i))
            pdf.savefig(fig)
            plt.close()
        pdf.close()


    def time_reparametrization(self,timepoints,xi,tau):
        return torch.exp(xi)* (timepoints - tau)

    def compute_sum_squared_tensorized(self, data, param_ind, MCMC):
        res = self.compute_individual_tensorized(data.timepoints, param_ind, MCMC)
        res *= data.mask
        return torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))

    def compute_individual_attachment_tensorized_mcmc(self, data, realizations):
        param_ind = self.get_param_from_real(realizations)
        attachment = self.compute_individual_attachment_tensorized(data, param_ind,MCMC=True)
        return attachment

    def compute_individual_attachment_tensorized(self,data,param_ind,MCMC):
        res = self.compute_individual_tensorized(data.timepoints, param_ind, MCMC)
        res *= data.mask
        squared_sum = torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))
        noise_var = self.parameters['noise_std'] ** 2
        attachment = 0.5 * (1 / noise_var) * squared_sum
        attachment += np.log(np.sqrt(2 * np.pi * noise_var))
        return attachment

    def update_model_parameters(self, data, suff_stats, burn_in_phase=True):
        # Memoryless part of the algorithm
        if burn_in_phase:
            self.update_model_parameters_burn_in(data, suff_stats)
        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            self.update_model_parameters_normal(data, suff_stats)
        self.attributes.update(['all'],self.parameters)

    def update_model_parameters_burn_in(self, data, realizations):
        raise NotImplementedError

    def get_population_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'population']

    def get_individual_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'individual']

    def __str__(self):
        output = "=== MODEL ===\n"
        for key in self.parameters.keys():
            output += "{0} : {1}\n".format(key, self.parameters[key])
        return output


    def compute_regularity_realization(self, realization):
        # Instanciate torch distribution
        if realization.variable_type == 'population':
            mean = self.parameters[realization.name]
            std = self.MCMC_toolbox['priors']['{0}_std'.format(realization.name)]
        elif realization.variable_type == 'individual':
            mean = self.parameters["{0}_mean".format(realization.name)]
            std = self.parameters["{0}_std".format(realization.name)]
        else:
            raise ValueError("Variable type not known")

        return self.compute_regularity_variable(realization.tensor_realizations,mean,std)

    def compute_regularity_variable(self,value,mean,std):
        # Instanciate torch distribution
        distribution = torch.distributions.normal.Normal(loc=mean,scale=std)
        return -distribution.log_prob(value)

    def get_realization_object(self, n_individuals):
        ### TODO : CollectionRealizations should probably get self.get_info_var rather than all self
        realizations = CollectionRealization()
        realizations.initialize(n_individuals, self)
        return realizations

    def random_variable_informations(self):
        raise NotImplementedError


    '''
    ###########################
    ## LEGACY
    ###########################
    
    def _update_random_variables(self):
        # TODO float for torch operations

        infos_variables = self.random_variable_informations()

        reals_pop_name = [infos_variables[key]["name"] for key in infos_variables.keys() if
                          infos_variables[key]["type"] == "population"]

        reals_ind_name = [infos_variables[key]["name"] for key in infos_variables.keys() if
                          infos_variables[key]["type"] == "individual"]

        for real_pop_name in reals_pop_name:
            self.random_variables[real_pop_name].mu = self.parameters[real_pop_name]

        for real_ind_name in reals_ind_name:
            self.random_variables[real_ind_name].mu = float(self.parameters["{0}_mean".format(real_ind_name)])
            self.random_variables[real_ind_name].variance = float(
                self.parameters["{0}_var".format(real_ind_name)])
    
    # Attachment
    def compute_attachment(self, data, reals_pop, reals_ind):
        return torch.stack([self.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices]).sum()

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return torch.stack([self.compute_individual_sumsquared(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices]).sum()

    def compute_individual_sumsquared(self, individual, reals_pop, real_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, real_ind)-individual.tensor_observations)**2)

    def compute_individual_attachment(self, individual, reals_pop, real_ind):
        #return self.compute_individual_sumsquared(individual, reals_pop, real_ind)*np.power(2*self.model_parameters['noise_var'], -1) + np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

        #TODO Remove constant terms ???
        constant_term = self.cache_variables['constant_fit_variable']
        noise_inverse = self.cache_variables['noise_inverse']

        sum_squared = self.compute_individual_sumsquared(individual, reals_pop, real_ind)

        fit = 0.5 * noise_inverse * sum_squared
        res = fit + constant_term
        return res

    def update_variable_info(self, key, reals_pop):
        """
        Check according to the key, if some intermediary parameters need to be re-computed.
        :param key:
        :return:
        """
        pass
    '''