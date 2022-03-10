import torch

from leaspy.algo.personalize.abstract_mcmc_personalize import AbstractMCMCPersonalizeAlgo
from leaspy.utils.typing import DictParamsTorch


class MeanReal(AbstractMCMCPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derived as the mean realization for `n_iter` samplings.

    TODO: harmonize naming in paths realiSation vs. realiZation...

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """
    name = 'mean_real'

    def _compute_individual_parameters_from_samples_torch(self,
            realizations: DictParamsTorch,
            attachments: torch.FloatTensor,
            regularities: torch.FloatTensor) -> DictParamsTorch:
        """
        Compute dictionary of individual parameters from stacked realizations, attachments and regularities.

        Parameters
        ----------
        realizations : dict[ind_var_name: str, `torch.FloatTensor` of shape (n_iter, n_individuals, *ind_var.shape)]
            The stacked history of realizations for individual variables.
        attachments : `torch.FloatTensor` of shape (n_iter, n_individuals)
            The stacked history of attachments.
        regularities : `torch.FloatTensor` of shape (n_iter, n_individuals)
            The stacked history of regularities (sum on all individual variables / dimensions).

        Returns
        -------
        dict[ind_var_name: str, `torch.FloatTensor` of shape (n_individuals, *ind_var.shape)]
        """
        # Only compute the mean of realizations (attachments & regularities not taken into account)
        return {
            ind_var_name: reals_var.mean(dim=0)
            for ind_var_name, reals_var in realizations.items()
        }
