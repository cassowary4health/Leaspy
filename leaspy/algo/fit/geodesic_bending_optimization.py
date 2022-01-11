import torch
import numpy as np

from . import kernel_utils
from ..abstract_algo import AbstractAlgo
import .kernel_utils

class GeodesicsBendingOptimization(AbstractAlgo):
    """
    Class with 'fit' algorithm for GeodesicsBending models uniquely.

    Attributes
    ----------
    current_iteration: int, default 0
        The number of the current iteration
    Inherited attributes
        From :class:`.AbstractAlgo`

    See also
    --------
    :meth:`.Leaspy.fit`
    """

    def __init__(self, settings):
        super().__init__()
        self.current_iteration = 0
        # Algorithm parameters
        self.algo_parameters = settings.parameters
        self.seed = settings.seed
        ####### WARNING : currently only handles MSE loss, needs more calculation in the future ###############
        self.loss = "MSE"

    def run(self, model, dataset):
        """
        Main method, run the algorithm.

        Basically, it solves a quadratic problem with quadratic constraint to optimize the kernel weights

        Parameters
        ----------
        model : :class:`~.models.geodesics_bending_model.GeodesicsBending`
            The used model.
        dataset : :class:`.Dataset`
            Contains the subjects' observations in torch format to speed up computation.

        Returns
        -------
        None

        """
        # Initialize algo_parameters
        if "sigma_auto" not in self.algo_parameters:
            self.algo_parameters["sigma_auto"] = False
        if "control_method" not in self.algo_parameters:
            self.algo_parameters["control_method"] = "grid"
        if self.algo_parameters["sigma_auto"]:
            sigma = kernel_utils.kernel_optimal_window(dataset)
            self.algo_parameters["sigma"] = sigma.detach().item()
        if "nb_control_points" in self.algo_parameters:
            k = self.algo_parameters["nb_control_points"]
#        if "iter" not in self.algo_parameters:
#            self.algo_parameters["iter"] = 200
#        if "iter_out_fit" not in meta_settings:
#            meta_settings["iter_out_fit"] = 400
        kernelsettings = {}
        kernelsettings["sigma"] = self.algo_parameters["sigma"]
        kernelsettings["kernel_name"] = self.algo_parameters["kernel_name"]
        if "order" in self.algo_parameters:
            kernelsettings["order"] = self.algo_parameters["order"]

        model.kernel_settings = kernelsettings

        # Estimation of base model individual parameters :
        # Base model should be calibrated already
        if "individual_parameters" in self.algo_parameters:
            individual_parameters = self.algo_parameters["individual_parameters"]
        elif "personalize_settings" in self.algo_parameters:
            individual_parameters = model.personalize(dataset, self.algo_parameters["personalize_settings"])
        else:
            raise ValueError("No way to compute individual parameters has been provided : either provide them directly\
                             with \"individual_parameters\" or provide settings for a personalize algorithm with \
                             \"personalize_settings\" in the algorithm parameters")

        _, ind_params = individual_parameters.to_pytorch()

        # Then we run the diffeomorphism estimation with the quadratic Optimization with update_b

        # We take the points in the base manifold
        X = self.model.compute_individual_tensorized(dataset.timepoints, ind_params)

        # Generation of control points
        if self.algo_parameters["control_method"] == "grid":
            control_points = kernel_utils.control_points_grid(dataset, self.algo_parameters["sigma"])
        elif self.algo_parameters["control_method"] == "subsampling":
            index = kernel_utils.sub_sampling(X, self.algo_parameters["nb_control_points"])
            control_points = X[index]
        else:
            raise ValueError("Methods to generate control points for kernel estimation are limited to \
                                     {} as of now".format(["grid", "subsampling"]))

        # Main optimization : see kernel_utils for the core of the optimization
        weights = kernel_utils.weights_optimization(X, dataset.values, self.algo_parameters, control_points)

        model.parameters["control_points"] = control_points
        model.parameters["weights"] = weights
        # Update the mapping associated to new weights and control points
        model.update_mapping()
        return

