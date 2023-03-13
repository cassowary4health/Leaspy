import torch
from abc import abstractmethod
from typing import Tuple

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import AbstractModel
from leaspy.io.data.dataset import Dataset
from leaspy.models.noise_models import AbstractGaussianNoiseModel


class AbstractPersonalizeAlgo(AbstractAlgo):
    """
    Abstract class for `personalize` algorithm.
    Estimation of individual parameters of a given `Data` file with
    a frozen model (already estimated, or loaded from known parameters).

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.

    Attributes
    ----------
    name : str
        Algorithm's name.
    seed : int, optional
        Algorithm's seed (default None).
    algo_parameters : dict
        Algorithm's parameters.

    See Also
    --------
    :meth:`.Leaspy.personalize`
    """

    family = 'personalize'

    def run_impl(self, model: AbstractModel, dataset: Dataset) -> Tuple[IndividualParameters, torch.FloatTensor]:
        r"""
        Main personalize function, wraps the abstract :meth:`._get_individual_parameters` method.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            A subclass object of leaspy `AbstractModel`.
        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        individual_parameters : :class:`.IndividualParameters`
            Contains individual parameters.
        noise_std : float or :class:`torch.FloatTensor`
            The estimated noise (is a tensor if `model.noise_model` is ``'gaussian_diagonal'``)

            .. math:: = \frac{1}{n_{visits} \times n_{dim}} \sqrt{\sum_{i, j \in [1, n_{visits}] \times [1, n_{dim}]} \varepsilon_{i,j}}

            where :math:`\varepsilon_{i,j} = \left( f(\theta, (z_{i,j}), (t_{i,j})) - (y_{i,j}) \right)^2` , where
            :math:`\theta` are the model's fixed effect, :math:`(z_{i,j})` the model's random effects,
            :math:`(t_{i,j})` the time-points and :math:`f` the model's estimator.
        """

        # Estimate individual parameters
        individual_parameters = self._get_individual_parameters(model, dataset)

        # Compute the noise with the estimated individual parameters (per feature or not, depending on model noise)
        _, individual_parameters = individual_parameters.to_pytorch()
        if isinstance(model.noise_model, AbstractGaussianNoiseModel):
            predictions = model.compute_individual_tensorized(
                dataset.timepoints, individual_parameters, attribute_type=None
            )
            loss = model.noise_model.compute_rmse(dataset, predictions)
        else:
            loss = model.compute_individual_attachment_tensorized(
                dataset, individual_parameters, attribute_type=None
            ).sum()

        return individual_parameters, loss

    @abstractmethod
    def _get_individual_parameters(self, model: AbstractModel, data: Dataset) -> IndividualParameters:
        """
        Estimate individual parameters from a `Dataset`.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            A subclass object of leaspy AbstractModel.
        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        :class:`.IndividualParameters`
        """
