from typing import Dict, Hashable, Union, Optional

import numpy as np
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.models.noise_models import (
    BaseNoiseModel,
    AbstractOrdinalNoiseModel,
    OrdinalRankingNoiseModel,
)


class OrdinalModelMixin:
    """Mix-in to add some useful properties & methods for models supporting the ordinal and ranking noise (univariate or multivariate)."""

    ## PUBLIC

    @property
    def is_ordinal(self) -> bool:
        """Property to check if the model is of ordinal sub-type."""
        if self.noise_model is None:
            return False
        return isinstance(self.noise_model, AbstractOrdinalNoiseModel)

    @property
    def ordinal_infos(self) -> Optional[dict]:
        if not self.is_ordinal:
            return None
        return self.noise_model.ordinal_infos

    def check_noise_model_compatibility(self, model: BaseNoiseModel) -> None:
        super().check_noise_model_compatibility(self, model)

        if isinstance(model, AbstractOrdinalNoiseModel) and self.name not in {'logistic', 'univariate_logistic'}:
            raise LeaspyModelInputError(
                "Noise model 'ordinal' is only compatible with 'logistic' and "
                f"'univariate_logistic' models, not {self.name}"
            )

    def postprocess_model_estimation(
            self,
            estimation: np.ndarray,
            *,
            ordinal_method: str = 'MLE',
            **kws,
    ) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        """
        Extra layer of processing used to output nice estimated values in main API `Leaspy.estimate`.

        Parameters
        ----------
        estimation : numpy.ndarray[float]
            The raw estimated values by model (from `compute_individual_trajectory`)
        ordinal_method : str
            <!> Only used for ordinal models.
            * 'MLE' or 'maximum_likelihood' returns maximum likelihood estimator for each point (int)
            * 'E' or 'expectation' returns expectation (float)
            * 'P' or 'probabilities' returns probabilities of all-possible levels for a given feature:
              {feature_name: array[float]<0..max_level_ft>}
        **kws
            Some extra keywords arguments that may be handled in the future.

        Returns
        -------
        numpy.ndarray[float] or dict[str, numpy.ndarray[float]]
            Post-processed values.
            In case using 'probabilities' mode, the values are a dictionary with keys being:
            `(feature_name: str, feature_level: int<0..max_level_for_feature>)`
            Otherwise it is a standard numpy.ndarray corresponding to different model features (in order)
        """
        if not self.is_ordinal:
            return estimation

        if isinstance(self.noise_model, OrdinalRankingNoiseModel):
            estimation = compute_ordinal_pdf_from_ordinal_sf(torch.tensor(estimation)).cpu().numpy()

        if ordinal_method in {'MLE', 'maximum_likelihood'}:
            return estimation.argmax(axis=-1)
        if ordinal_method in {'E', 'expectation'}:
            return np.flip(estimation, axis=-1).cumsum(axis=-1).sum(axis=-1) - 1.
        if ordinal_method in {'P', 'probabilities'}:
            d_ests = {}
            for ft_i, feat in enumerate(self.noise_model.features):
                for ft_lvl in range(feat["max_level"] + 1):
                    d_ests[(feat["name"], ft_lvl)] = estimation[..., ft_i, ft_lvl]
            return d_ests
        raise LeaspyInputError(
            "`ordinal_method` should be in: {'maximum_likelihood', 'MLE', "
            "'expectation', 'E', 'probabilities', 'P'} "
            f"not {ordinal_method}."
        )

    ## PRIVATE

    def _ordinal_grid_search_value(self, grid_timepoints: torch.Tensor, values: torch.Tensor, *,
                                   individual_parameters: Dict[str, torch.Tensor], feat_index: int) -> torch.Tensor:
        """Search first timepoint where ordinal MLE is >= provided values."""
        grid_model = self.compute_individual_tensorized_logistic(
            grid_timepoints.unsqueeze(0), individual_parameters, attribute_type=None
        )[:, :, [feat_index], :]

        if isinstance(self.noise_model, OrdinalRankingNoiseModel):
            grid_model = compute_ordinal_pdf_from_ordinal_sf(grid_model)

        # we search for the very first timepoint of grid where ordinal MLE was >= provided value
        # TODO? shouldn't we return the timepoint where P(X = value) is highest instead?
        MLE = grid_model.squeeze(dim=2).argmax(dim=-1) # squeeze feat_index (after computing pdf when needed)
        index_cross = (MLE.unsqueeze(1) >= values.unsqueeze(-1)).int().argmax(dim=-1)

        return grid_timepoints[index_cross]

    @property
    def _attributes_factory_ordinal_kws(self) -> dict:
        # we put this here to remain more generic in the models
        return dict(ordinal_infos=self.ordinal_infos)

    def _export_extra_ordinal_settings(self, model_settings) -> None:
        if self.is_ordinal:
            model_settings['batch_deltas_ordinal'] = self.noise_model.batch_deltas

    def _handle_ordinal_hyperparameters(self, hyperparameters) -> tuple:
        """Return a tuple of extra hyperparameters that are recognized."""
        if not self.is_ordinal:
            return tuple()  # no extra hyperparameters recognized

        self.noise_model.batch_deltas = hyperparameters.get('batch_deltas_ordinal', False)

        return ('batch_deltas_ordinal',)

    def _initialize_MCMC_toolbox_ordinal_priors(self) -> None:
        if not self.is_ordinal:
            return
        if self.noise_model.batch_deltas:
            self.MCMC_toolbox['priors']['deltas_std'] = 0.1
        else:
            for feat in self.noise_model.features:
                self.MCMC_toolbox['priors'][f'deltas_{feat["name"]}_std'] = 0.1

    def _update_MCMC_toolbox_ordinal(self, vars_to_update: set, realizations, values: dict) -> None:
        # update `values` dict in-place
        if not self.is_ordinal:
            return
        update_all = 'all' in vars_to_update
        if self.noise_model.batch_deltas:
            if update_all or 'deltas' in vars_to_update:
                values['deltas'] = realizations['deltas'].tensor_realizations
        else:
            for feat in self.noise_model.features:
                if update_all or 'deltas_'+feat["name"] in vars_to_update:
                    values['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

    def get_ordinal_tensor_realizations(self, realizations) -> dict:
        if not self.is_ordinal:
            return {}
        if self.noise_model.batch_deltas:
            return {"deltas": realizations['deltas'].tensor_realizations}
        return {
            f"deltas_{feat['name']}": realizations[f"deltas_{feat['name']}"].tensor_realizations
            for feat in self.noise_model.features
        }

    def get_ordinal_sufficient_statistics(self, sufficient_statistics: dict) -> dict:
        if not self.is_ordinal:
            return {}
        if self.noise_model.batch_deltas:
            return {"deltas": sufficient_statistics["deltas"]}
        return {
            f"deltas_{feat['name']}": sufficient_statistics[f"deltas_{feat['name']}"]
            for feat in self.noise_model.features
        }

    def _get_deltas(self, attribute_type: Optional[str]) -> torch.Tensor:
        """
        Get the deltas attribute for ordinal models.

        Parameters
        ----------
        attribute_type: None or 'MCMC'

        Returns
        -------
        The deltas in the ordinal model
        """
        return self._call_method_from_attributes('get_deltas', attribute_type)

    def _add_ordinal_random_variables(self, variables_infos: dict) -> None:
        if not self.is_ordinal:
            return
        deltas_info = {"type": "population", "scale": .5}
        if self.noise_model.batch_deltas:
            max_level = self.noise_model.max_level
            deltas_info.update(
                {
                    "name": "deltas",
                    "shape": torch.Size([self.dimension, max_level - 1]),
                    "rv_type": "multigaussian",
                    "mask": self.noise_model.mask[0, 0, :, 1:],  # cut the zero level
                }
            )
            variables_infos['deltas'] = deltas_info
        else:
            for feat in self.noise_model.features:
                deltas_info.update(
                    {
                        "name": "deltas_" + feat["name"],
                        "shape": torch.Size([feat["max_level"] - 1]),
                        "rv_type": "gaussian",
                    }
                )
                variables_infos['deltas_' + feat["name"]] = deltas_info

        # Finally: change the v0 scale since it has not the same meaning
        if 'v0' in variables_infos:
            variables_infos['v0']['scale'] = 0.1


def compute_ordinal_pdf_from_ordinal_sf(
    ordinal_sf: torch.Tensor,
    dim_ordinal_levels: int = 3,
) -> torch.Tensor:
    """
    Computes the probability density (or its jacobian) of an ordinal
    model [P(X = l), l=0..L] from `ordinal_sf` which are the survival
    function probabilities [P(X > l), i.e. P(X >= l+1), l=0..L-1] (or its jacobian).

    Parameters
    ----------
    ordinal_sf : `torch.FloatTensor`
        Survival function values : ordinal_sf[..., l] is the proba to be superior or equal to l+1
        Dimensions are:
        * 0=individual
        * 1=visit
        * 2=feature
        * 3=ordinal_level [l=0..L-1]
        * [4=individual_parameter_dim_when_gradient]
    dim_ordinal_levels : int, default = 3
        The dimension of the tensor where the ordinal levels are.

    Returns
    -------
    ordinal_pdf : `torch.FloatTensor` (same shape as input, except for dimension 3 which has one more element)
        ordinal_pdf[..., l] is the proba to be equal to l (l=0..L)
    """
    # nota: torch.diff was introduced in v1.8 but would not highly improve performance of this routine anyway
    s = list(ordinal_sf.shape)
    s[dim_ordinal_levels] = 1
    last_row = torch.zeros(size=tuple(s))
    if len(s) == 5:  # in the case of gradient we added a dimension
        first_row = last_row  # gradient(P>=0) = 0
    else:
        first_row = torch.ones(size=tuple(s))  # (P>=0) = 1
    sf_sup = torch.cat([first_row, ordinal_sf], dim=dim_ordinal_levels)
    sf_inf = torch.cat([ordinal_sf, last_row], dim=dim_ordinal_levels)
    pdf = sf_sup - sf_inf

    return pdf


def compute_ordinal_sf_from_ordinal_pdf(ordinal_pdf: Union[torch.Tensor, np.ndarray]):
    """
    Compute the ordinal survival function values [P(X > l), i.e.
    P(X >= l+1), l=0..L-1] (l=0..L-1) from the ordinal probability density
    [P(X = l), l=0..L] (assuming ordinal levels are in last dimension).
    """
    return (1 - ordinal_pdf.cumsum(-1))[..., :-1]
    #return backend.flip(backend.flip(ordinal_pdf, (-1,)).cumsum(-1), (-1,))[..., 1:] # also correct
