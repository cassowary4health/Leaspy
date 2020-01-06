import torch
from leaspy.utils.realizations.collection_realization import CollectionRealization
import math

from leaspy.utils.realizations.realization import Realization

TWO_PI = 2 * math.pi


# TODO: Check & complete docstrings
class AbstractModel:
    """
    AbstractModel class contains the common attributes & methods of the different models.

    Attributes
    ----------
    distribution: torch.distributions.normal.Normal class object
        Gaussian generator for the model's penalty (?)
    is_initialized: bool
        Indicates if the model is initialized
    name: str
        The model's name
    parameters: dict
        Contains the model's parameters

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
    """

    def __init__(self, name):
        self.is_initialized = False
        self.name = name
        self.features = None
        self.parameters = None
        self.distribution = torch.distributions.normal.Normal(loc=0., scale=0.)

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

    def load_hyperparameters(self, hyperparameters):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def get_individual_variable_name(self):
        """
        Return list of names of the individual variables from the model.

        Returns
        -------
        list
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
        TODO: complete
        Compute the square of the residuals. (?) from one subject? Several subjects? All subject?

        Parameters
        ----------
        data: a leaspy.inputs.data.dataset.Dataset class object
            Contains the data of the subjects, in particular the subjects' time-points and the mask (?)
        param_ind: dict
            Contain the individual parameters
        attribute_type: str
            The attribute's type

        Returns
        -------
        torch.Tensor
            Contain one residuals for each subject? Visit? Sub-score?
        """
        res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        return torch.sum((res * data.mask.float() - data.values) ** 2, dim=(1, 2))

    # TODO: unit & functional tests
    def compute_individual_trajectory(self, timepoints, individual_parameters):
        """
        Compute the individual scores' values of a subject given his individual parameters at the given time-point(s).

        Parameters
        ----------
        timepoints: scalar or list
            Contains the age(s) of the subjects.
        individual_parameters: dict
            Contains the individual parameters.

        Returns
        -------
        torch.Tensor
            Contains the subject's scores computed at the given age(s)
        """

        # Check the given individual parameters' names & convert them to torch tensor
        available_parameters = ['xi', 'tau'] + (self.name != 'univariate') * ['sources']
        for key in individual_parameters.keys():
            assert key in available_parameters,\
                'The individual parameter {} is not available for {} model! ' \
                'The available individual parameters are {}.'.\
                format(key, self.name, available_parameters)

            if type(individual_parameters[key]) == torch.Tensor:
                continue

            if type(individual_parameters[key]) != list:
                individual_parameters[key] = [individual_parameters[key]]
            individual_parameters[key] = torch.tensor(individual_parameters[key], dtype=torch.float32).unsqueeze(0)

        # Convert the timepoints (list of numbers, or single number) to a torch tensor
        if type(timepoints) != list:
            timepoints = [timepoints]
        timepoints = torch.tensor(timepoints, dtype=torch.float32).unsqueeze(0)

        # Compute the individual trajectory
        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def compute_individual_tensorized(self, timepoints, individual_parameters, attribute_type=None):
        return NotImplementedError

    def compute_individual_attachment_tensorized_mcmc(self, data, realizations):
        """
        TODO: complete
        Compute attachment of all subjects? One subject? One visit?

        Parameters
        ----------
        data: a leaspy.inputs.data.dataset.Dataset class object
            Contains the data of the subjects, in particular the subjects' time-points and the mask (?)
        realizations: a leaspy realization class object

        Returns
        -------
        scalar
            The subject attachment (?)
        """
        param_ind = self.get_param_from_real(realizations)
        attachment = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type='MCMC')
        return attachment

    def compute_individual_attachment_tensorized(self, data, param_ind, attribute_type):
        """
        TODO: complete
        Compute attachment of all subjects? One subject? One visit?

        Parameters
        ----------
        data: a leaspy.inputs.data.dataset.Dataset class object
            Contains the data of the subjects, in particular the subjects' time-points and the mask (?)
        param_ind
        attribute_type: str

        Returns
        -------

        """
        res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        # res *= data.mask

        r1 = res * data.mask.float() - data.values
        #r1[1-data.mask] = 0.0 # Set nans to 0
        squared_sum = torch.sum(r1 * r1, dim=(1, 2))

        # noise_var = self.parameters['noise_std'] ** 2
        noise_var = self.parameters['noise_std'] * self.parameters['noise_std']
        attachment = 0.5 * (1 / noise_var) * squared_sum

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

    def update_model_parameters_burn_in(self, data, realizations):
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
        # Instanciate torch distribution
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
        # TODO change to static ???
        # Instanciate torch distribution
        # distribution = torch.distributions.normal.Normal(loc=mean, scale=std)

        self.distribution.loc = mean
        self.distribution.scale = std

        return -self.distribution.log_prob(value)

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

    def time_reparametrization(self, timepoints, xi, tau):
        return torch.exp(xi) * (timepoints - tau)

    def get_param_from_real(self, realizations):

        individual_parameters = dict.fromkeys(self.get_individual_variable_name())

        for variable_ind in self.get_individual_variable_name():
            if variable_ind == "sources" and self.source_dimension == 0:
                individual_parameters[variable_ind] = None
            else:
                individual_parameters[variable_ind] = realizations[variable_ind].tensor_realizations

        return individual_parameters
