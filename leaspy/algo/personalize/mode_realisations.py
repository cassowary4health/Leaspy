import torch

from leaspy.algo.personalize.abstract_mcmc_personalize import AbstractMCMCPersonalizeAlgo
from leaspy.utils.typing import DictParamsTorch


class ModeReal(AbstractMCMCPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derived as the most frequent realization for `n_iter` samplings.

    TODO? we could derive some confidence intervals on individual parameters thanks to this personalization algorithm...

    TODO: harmonize naming in paths realiSation vs. realiZation...

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """
    name = 'mode_real'

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
        # Indices of iterations where loss (= negative log-likelihood) was minimal
        # (per individual, but tradeoff on ALL individual parameters)
        indices_iter_best = torch.argmin(attachments + regularities, dim=0)  # shape (n_individuals,)

        return {
            ind_var_name: torch.stack([reals_var[ind_best_iter, ind]
                                       for ind, ind_best_iter in enumerate(indices_iter_best)])
            for ind_var_name, reals_var in realizations.items()
        }
