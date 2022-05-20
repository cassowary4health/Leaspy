from typing import Dict, Hashable, Union

import numpy as np
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError


class OrdinalModelMixin:
    """Mix-in to add some useful properties & methods for models supporting the ordinal and ranking noise (univariate or multivariate)."""

    ## PUBLIC

    @property
    def is_ordinal(self) -> bool:
        """Property to check if the model is of ordinal sub-type."""
        return self.noise_model in ['ordinal', 'ordinal_ranking']

    def postprocess_model_estimation(self, estimation: np.ndarray, *, ordinal_method: str = 'MLE', **kws) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
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

        if self.noise_model == 'ordinal_ranking':
            # start by computing pdf from sf
            estimation = self.compute_ordinal_pdf_from_ordinal_sf(torch.tensor(estimation)).cpu().numpy()

        # postprocess the ordinal pdf depending on `ordinal_method`
        if ordinal_method in {'MLE', 'maximum_likelihood'}:
            return estimation.argmax(axis=-1)
        elif ordinal_method in {'E', 'expectation'}:
            return np.flip(estimation, axis=-1).cumsum(axis=-1).sum(axis=-1) - 1.
        elif ordinal_method in {'P', 'probabilities'}:
            # we construct a dictionary with the appropriate keys
            d_ests = {}
            for ft_i, feat in enumerate(self.ordinal_infos["features"]):
                for ft_lvl in range(0, feat["max_level"] + 1):
                    d_ests[(feat["name"], ft_lvl)] = estimation[..., ft_i, ft_lvl]

            return d_ests
        else:
            raise LeaspyInputError("`ordinal_method` should be in: {'maximum_likelihood', 'MLE', 'expectation', 'E', 'probabilities', 'P'}"
                                   f" not {ordinal_method}")


    def compute_ordinal_pdf_from_ordinal_sf(self, ordinal_sf: torch.Tensor, *, dim_ordinal_levels: int = 3) -> torch.Tensor:
        """
        Computes the probability density (or its jacobian) of an ordinal model [P(X = l), l=0..L] from `ordinal_sf` which are the survival function probabilities [P(X > l), i.e. P(X >= l+1), l=0..L-1] (or its jacobian).

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

    @staticmethod
    def compute_ordinal_sf_from_ordinal_pdf(ordinal_pdf: Union[torch.Tensor, np.ndarray]):
        """Compute the ordinal survival function values [P(X > l), i.e. P(X >= l+1), l=0..L-1] (l=0..L-1) from the ordinal probability density [P(X = l), l=0..L] (assuming ordinal levels are in last dimension)."""
        return (1 - ordinal_pdf.cumsum(-1))[..., :-1]
        #return backend.flip(backend.flip(ordinal_pdf, (-1,)).cumsum(-1), (-1,))[..., 1:] # also correct

    ## PRIVATE

    def _ordinal_grid_search_value(self, grid_timepoints: torch.Tensor, values: torch.Tensor, *,
                                   individual_parameters: Dict[str, torch.Tensor], feat_index: int) -> torch.Tensor:
        """Search first timepoint where ordinal MLE is >= provided values."""
        grid_model = self.compute_individual_tensorized_logistic(grid_timepoints.unsqueeze(0), individual_parameters,
                                                                 attribute_type=None)[:,:,[feat_index],:]

        if self.noise_model == 'ordinal_ranking':
            grid_model = self.compute_ordinal_pdf_from_ordinal_sf(grid_model)

        # we search for the very first timepoint of grid where ordinal MLE was >= provided value
        # TODO? shouldn't we return the timepoint where P(X = value) is highest instead?
        MLE = grid_model.squeeze(dim=2).argmax(dim=-1) # squeeze feat_index (after computing pdf when needed)
        index_cross = (MLE.unsqueeze(1) >= values.unsqueeze(-1)).int().argmax(dim=-1)

        return grid_timepoints[index_cross]


    @property
    def _attributes_factory_ordinal_kws(self) -> dict:
        # we put this here to remain more generic in the models
        return dict(ordinal_infos=getattr(self, 'ordinal_infos',  None))

    def _export_extra_ordinal_settings(self, model_settings) -> None:

        if self.is_ordinal:
            model_settings['batch_deltas_ordinal'] = self.ordinal_infos["batch_deltas"]

    def _handle_ordinal_hyperparameters(self, hyperparameters) -> tuple:
        # return a tuple of extra hyperparameters that are recognized

        if not self.is_ordinal:
            return tuple()  # no extra hyperparameters recognized

        if self.name not in {'logistic', 'univariate_logistic'}:
            raise LeaspyModelInputError(f"Noise model 'ordinal' is only compatible with 'logistic' and 'univariate_logistic' models, not {self.name}")

        if hasattr(self, 'ordinal_infos'):
            self.ordinal_infos["batch_deltas"] = hyperparameters.get('batch_deltas_ordinal',
                                                                     self.ordinal_infos["batch_deltas"])
        else:
            # initialize the ordinal_infos dictionary
            self.ordinal_infos = {"batch_deltas": hyperparameters.get('batch_deltas_ordinal', False),
                                  "max_level": None,
                                  "features": [],
                                  "mask": None,
                                 }

        return ('batch_deltas_ordinal',)

    def _initialize_MCMC_toolbox_ordinal_priors(self) -> None:

        if not self.is_ordinal:
            return

        if self.ordinal_infos['batch_deltas']:
            self.MCMC_toolbox['priors']['deltas_std'] = 0.1
        else:
            for feat in self.ordinal_infos["features"]:
                self.MCMC_toolbox['priors'][f'deltas_{feat["name"]}_std'] = 0.1

    def _update_MCMC_toolbox_ordinal(self, vars_to_update: tuple, realizations, values: dict) -> None:
        # update `values` dict in-place

        if not self.is_ordinal:
            return

        update_all = 'all' in vars_to_update
        if self.ordinal_infos['batch_deltas']:
            if update_all or 'deltas' in vars_to_update:
                values['deltas'] = realizations['deltas'].tensor_realizations
        else:
            for feat in self.ordinal_infos["features"]:
                if update_all or 'deltas_'+feat["name"] in vars_to_update:
                    values['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

    def _add_ordinal_tensor_realizations(self, realizations, dict_to_update: dict) -> None:
        if not self.is_ordinal:
            return

        if self.ordinal_infos['batch_deltas']:
            dict_to_update['deltas'] = realizations['deltas'].tensor_realizations
        else:
            for feat in self.ordinal_infos["features"]:
                dict_to_update['deltas_'+feat["name"]] = realizations['deltas_'+feat["name"]].tensor_realizations

    def _add_ordinal_sufficient_statistics(self, suff_stats: dict, dict_to_update: dict) -> None:
        if not self.is_ordinal:
            return

        # The only difference with `_add_ordinal_tensor_realizations` is that suff_stats is a dict of Tensors and not
        # a CollectionRealizations object (which needs) to fetch the `.tensor_realizations` attribute...
        if self.ordinal_infos['batch_deltas']:
            dict_to_update['deltas'] = suff_stats['deltas']
        else:
            for feat in self.ordinal_infos["features"]:
                dict_to_update['deltas_'+feat["name"]] = suff_stats['deltas_'+feat["name"]]


    def _rebuild_ordinal_infos_from_model_parameters(self) -> None:

        # is this an ordinal model?
        if not self.is_ordinal:
            return

        # if yes: re-build the number of levels per feature
        deltas_p = {k: v for k, v in self.parameters.items() if k.startswith('deltas')}

        if self.ordinal_infos['batch_deltas']:
            assert deltas_p.keys() == {'deltas'}
            # Find ordinal infos from the delta values themselves
            undef_levels = torch.isinf(self.parameters['deltas'])
            self.ordinal_infos["max_level"] = undef_levels.shape[1] + 1
            for i, feat in enumerate(self.features):
                undef_levels_ft = undef_levels[i, :]
                if undef_levels_ft.any():
                    max_lvl_ft = undef_levels_ft.int().argmax().item() + 1
                else:
                    max_lvl_ft = self.ordinal_infos["max_level"]
                self.ordinal_infos["features"].append({"name": feat, "max_level": max_lvl_ft})
        else:
            assert deltas_p.keys() == {f'deltas_{ft}' for ft in self.features}
            for k, v in deltas_p.items():
                feat = k[7:]  #k[7:] removes the deltas_ to extract the feature's name
                self.ordinal_infos["features"].append({"name": feat, "max_level": v.shape[0] + 1})

            self.ordinal_infos["max_level"] = max([feat["max_level"] for feat in self.ordinal_infos["features"]])

        # re-build the mask to account for possible difference in levels per feature
        # shape of mask is (1, 1, dimension, ordinal_max_level)
        self.ordinal_infos["mask"] = torch.cat([
            torch.cat([
                torch.ones((1,1,1,feat['max_level'])),
                torch.zeros((1,1,1,self.ordinal_infos['max_level'] - feat['max_level'])),
            ], dim=-1) for feat in self.ordinal_infos["features"]
        ], dim=2)

    def _get_deltas(self, attribute_type: str) -> torch.Tensor:
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

        if self.ordinal_infos['batch_deltas']:
            # Instead of a sampler for each feature, sample deltas for all features in one sampler class
            max_level = self.ordinal_infos["max_level"]
            deltas_infos = {
                "name": "deltas",
                "shape": torch.Size([self.dimension, max_level - 1]),
                "type": "population",
                "rv_type": "multigaussian",
                "scale": .5,
                "mask": self.ordinal_infos["mask"][0,0,:,1:], # cut the zero level
            }
            variables_infos['deltas'] = deltas_infos
        else:
            # For each feature : create a sampler for deltas of size (max_level_of_the_feature - 1)
            for feat in self.ordinal_infos["features"]:
                deltas_infos = {
                    "name": "deltas_"+feat["name"],
                    "shape": torch.Size([feat["max_level"] - 1]),
                    "type": "population",
                    "rv_type": "gaussian",
                    "scale": .5,
                }
                variables_infos['deltas_'+feat["name"]] = deltas_infos

        # Finally: change the v0 scale since it has not the same meaning
        if 'v0' in variables_infos:  # not in univariate case!
            variables_infos['v0']['scale'] = 0.1
