import math
import time

import torch

from ..abstract_algo import AbstractAlgo


class AbstractPersonalizeAlgo(AbstractAlgo):
    """
    Abstract class for "personalize" algorithm.
    Estimation of individual parameters of a given "Data" file with
    a freezed model (already estimated, or loaded from known parameters)

    Attributes
    ----------
    algo_parameters: str
        algorithm's name
     name: str
        algorithm's name
     seed: int
        algorithm's seed (default None)

    Methods
    -------
    _get_individual_parameters(model, data)
        Estimate individual parameters from a data using leaspy object class model & Dataset
    run(model, data)
        Main personalize function, wraps the _get_individual_parameters
    """

    def __init__(self, settings):
        """
        Initialize class object from settings object

        Parameters
        ----------
        settings : leaspy.io.settings.algorithm_settings.AlgorithmSettings
            Settings of the algorithm
        """
        super().__init__()

        # Algorithm parameters
        self.algo_parameters = settings.parameters

        # Name
        self.name = settings.name

        # Seed
        self.seed = settings.seed
        self.loss = settings.loss

    def run(self, model, data):
        """
        Main personalize function, wraps the _get_individual_parameters

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            A subclass object of leaspy AbstractModel.
        data : leaspy.io.data.dataset.Dataset
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        tuple (ips, noise_std)

        ips: dict
            individual parameters
            example {'xi': <list of float>, 'tau': <list of float>, 'sources': <list of list of float>}

        noise_std: torch.FloatTensor of shape (dimension,) or scalar
            estimated noise (per feature or globally)
        """

        # Set seed
        self._initialize_seed(self.seed)

        # Init the run
        #print("Beginning personalization : std error of the model is {0}".format(model.parameters['noise_std']))
        time_beginning = time.time()

        ## Give the model the adequate loss
        #model.loss = self.loss

        # mod Etienne: during personalization we should rather do the opposite (algorithm loss must correspond to model loss, for wich the model was trained for)
        self.loss = model.loss

        # Estimate individual parametersabstr
        individual_parameters = self._get_individual_parameters(model, data)

        # Compute the noise with the estimated individual parameters (per feature or not, depending on model loss)
        _, dict_pytorch = individual_parameters.to_pytorch()
        if 'diag_noise' in model.loss:
            squared_diff = model.compute_sum_squared_per_ft_tensorized(data, dict_pytorch).sum(dim=0) # k tensor
            noise_std = torch.sqrt(squared_diff.detach() / data.n_observations_per_ft.float())

            # for displaying only
            noise_map = {ft_name: '{:.4f}'.format(ft_noise) for ft_name, ft_noise in zip(model.features, noise_std)}
            print_noise = repr(noise_map).replace("'", "")
        else:
            squared_diff = model.compute_sum_squared_tensorized(data, dict_pytorch).sum()
            noise_std = torch.sqrt(squared_diff.detach() / data.n_observations)
            # for displaying only
            print_noise = '{:.4f}'.format(noise_std.item())

        # Print run infos
        time_end = time.time()
        diff_time = (time_end - time_beginning)

        print("The standard deviation of the noise at the end of the personalization is " + print_noise)
        print("Personalization {} took : {:.1f}s".format(self.name, diff_time))

        return individual_parameters, noise_std

    def _get_individual_parameters(self, model, data):
        """
        Estimate individual parameters from a data using leaspy object class model & Dataset

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            A subclass object of leaspy AbstractModel.
        data : leaspy.io.data.dataset.Dataset
            Dataset object build with leaspy class objects Data, algo & model

        Raises
        ------
        NotImplementedError
            Method only implemented in child class of AbstractPersonalizeAlgo

        Should return
        ------
        leaspy.io.outputs.individual_parameters.IndividualParameters
        """

        raise NotImplementedError('This algorithm does not present a personalization procedure')
