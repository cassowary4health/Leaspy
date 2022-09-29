import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus

from leaspy.exceptions import LeaspyModelInputError

available_types = ["gaussian_kernel"]

class TreatmentFunction():
    """
    Creates a treatment function of the chosen type parametrized by weights.

    Parameters
    __________
    dimension : int
        The number of features of the current model
    function_type : str
        The class of the function to model disease-modifying treatment effect.
        Currently only 'gaussian_kernel' is implemented.
    initial_weights : torch.tensor
        Optional weights to initialize the function. Otherwise it initializes at zero.
    """

    def __init__(self, dimension=None, data=None, function_type='gaussian_kernel', initial_weights=None, **kwargs):
        if dimension is None:
            if data is None:
                raise LeaspyModelInputError("Dimension or data is required to initialize TreatmentFunction")
            else:
                dimension = data.shape[1]
        self.dimension = dimension
        self.type = function_type
        if self.type not in available_types:
            raise LeaspyModelInputError(f'Only {available_types} treatment function is implemented as of now.')
        if self.type == 'gaussian_kernel':
            self.nb_control_points = kwargs.get("nb_control_points", 100)
            self.control_points = kwargs.get("control_points", None)
            if self.control_points is None:
                if data is None:
                    raise LeaspyModelInputError("TreatmentFunction of 'gaussian_kernel' type requires either control_points or data argument in order to work")
                else:
                    self.create_control_points(data)
            self.sigma = kwargs.get('sigma', 1.)
            self.regularization = kwargs.get('regularization', 'L1')
            self.regularization_weight = kwargs.get('regularization_weight', 1.)
        if initial_weights is None:
            if self.type == 'gaussian_kernel':
                initial_weights = torch.zeros((self.nb_control_points, dimension))
        self.weights = initial_weights

    #TODO suffix method ?
    def __call__(self, p, v, covariates=None):
        """
        Applies the function to the states defined by the pair (p, v) with optional covariates
        Parameters
        ----------
        p : torch.tensor
            Positions
        v : torch.tensor
            Velocities
        covariates
            Optional covariates, not yet taken into account
        Returns
        -------
        The new velocities after the function is applied to the pair (p,v)
        """
        if self.type == 'gaussian_kernel':
            return self.apply_gaussian_kernel(p, v, covariates=covariates)

    def apply_gaussian_kernel(self, p, v, covariates=None):
        """
        Applies the function to the states defined by the pair (p, v) with optional covariates
        Parameters
        ----------
        p : torch.tensor
            Positions
        v : torch.tensor
            Velocities
        covariates
            Optional covariates, not yet taken into account
        Returns
        -------
        The new velocities after the function is applied to the pair (p,v)
        """
        centers = self.control_points
        W = self.weights
        sigma =self.sigma

        size = p.shape[0]

        c = torch.tile(centers.unsqueeze(0), (size, 1, 1))
        p = torch.tile(p.unsqueeze(-1), (1, self.nb_control_points, 1))
        # Computing Gaussian kernel between centroids and positions p
        K = c - p
        sig = sigma * sigma * 2.
        K = (K * K / sig).sum(dim=-1)
        K = torch.exp(-K)
        return v + K.matmul(W)

    def create_control_points(self, p, random_state=42):
        """
        Creates control points according to Kmeans + +
        Parameters
        ----------
        p : torch.tensor
            The values of the data
        random_state : int
            Seed for random generator
        Returns
        -------
        None
        """
        centers, indices = kmeans_plusplus(p.numpy(), n_clusters=self.nb_control_points, random_state=random_state)
        self.control_points = p[indices]

    def optimization_step(self, model, dataset, ip, alpha=0.1, beta=0.7):
        """
        Operates a backtracking line search step in the gradient direction.
        Parameters
        ----------
        model : Leaspy.class.model
            Model to compute attachment loss from which to compute the gradient, not including the regularization.
        dataset : Leaspy.Dataset
            Data
        ip : Leaspy.IndividualParameters
        Returns
        -------
        None
        """

        #TODO optimize the computation of gradient steps

        def compute_loss(dataset, model, ip):
            values = model.compute_individual_tensorized(dataset.timepoints, ip, attribute_type='MCMC')
            if model.loss in ['gaussian_scalar', 'gaussian_diagonal']:
                return torch.nn.functional.mse_loss(values, dataset.values)

        def backtracking_line_search(eval_function, direction, param, max_step=1., alpha=0.1, beta=0.7):
            x0 = param
            current_val = eval_function(x0)
            t = max_step
            x = x0 - t * direction
            while eval_function(x) - current_val >= - alpha * t * (direction * direction).sum():
                t = beta * t
                x = x0 - t * direction
            return x, t

        def eval_function(x):
            self.weights = x
            with torch.no_grad():
                loss = compute_loss(dataset, model, ip)
            return loss

        self.weights.requires_grad = True
        self.weights.grad = None
        loss = compute_loss(dataset, model, ip)
        # Regularization
        if self.regularization == 'L1':
            reg = torch.norm(self.weights, 1)
        elif self.regularization == 'L2':
            reg = torch.norm(self.weights, 2)
        loss = loss + reg
        loss.backward()
        grad = self.weights.grad
        better_param, step = backtracking_line_search(eval_function, grad, self.weights, alpha=alpha, beta=beta)
        self.weights = better_param.detach().clone() #clone to recreate Torch graph for gradient computation
