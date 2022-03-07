from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.typing import List

if TYPE_CHECKING:
    from leaspy.io.data.data import Data
    from leaspy.models.abstract_model import AbstractModel
    from leaspy.algo.abstract_algo import AbstractAlgo


class Dataset:
    """
    Data container based on :class:`torch.Tensor`, used to run algorithms.

    Parameters
    ----------
    data : :class:`.Data`
        Create `Dataset` from `Data` object
    model : :class:`.AbstractModel` (optional)
        If not None, will check compatibility of model and data
    algo : :class:`.AbstractAlgo` (optional)
        If not None, will check compatibility of algo and data

    Attributes
    ----------
    headers : list[str]
        Features names
    dimension : int
        Number of features
    n_individuals : int
        Number of individuals
    indices : list[ID]
        Order of patients

    n_visits_per_individual : list[int]
        Number of visits per individual
    n_visits_max : int
        Maximum number of visits for one individual
    n_visits : int
        Total number of visits

    n_observations_per_ind_per_ft : :class:`torch.LongTensor`, shape (n_individuals, dimension)
        Number of observations (not taking into account missing values) per individual per feature
    n_observations_per_ft : :class:`torch.LongTensor`, shape (dimension,)
        Total number of observations per feature
    n_observations : int
        Total number of observations

    timepoints : :class:`torch.FloatTensor`, shape (n_individuals, n_visits_max)
        Ages of patients at their different visits
    values : :class:`torch.FloatTensor`, shape (n_individuals, n_visits_max, dimension)
        Values of patients for each visit for each feature
    mask : :class:`torch.FloatTensor`, shape (n_individuals, n_visits_max, dimension)
        Binary mask associated to values.
        If 1: value is meaningful
        If 0: value is meaningless (either was nan or does not correspond to a real visit - only here for padding)

    L2_norm_per_ft : :class:`torch.FloatTensor`, shape (dimension,)
        Sum of all non-nan squared values, feature per feature
    L2_norm : scalar :class:`torch.FloatTensor`
        Sum of all non-nan squared values

    Raises
    ------
    :exc:`.LeaspyInputError`
        if data, model or algo are not compatible together.
    """

    def __init__(self, data: Data, model: AbstractModel = None, algo: AbstractAlgo = None):

        self.headers = data.headers
        self.dimension = data.dimension
        self.n_individuals = data.n_individuals
        self.n_visits = data.n_visits
        self.indices = list(data.individuals.keys())

        self.timepoints: torch.FloatTensor = None
        self.values: torch.FloatTensor = None
        self.mask: torch.FloatTensor = None

        self.n_observations: int = None
        self.n_observations_per_ft: torch.LongTensor = None
        self.n_observations_per_ind_per_ft: torch.LongTensor = None

        self.n_visits_per_individual: List[int] = None
        self.n_visits_max: int = None

        self.L2_norm_per_ft: torch.FloatTensor = None
        self.L2_norm: torch.FloatTensor = None

        if model is not None:
            self._check_model_compatibility(data, model)
        if algo is not None:
            self._check_algo_compatibility(data, algo)

        self._construct_values(data)
        self._construct_timepoints(data)
        self._compute_L2_norm()

    def _construct_values(self, data: Data):

        self.n_visits_per_individual = [len(_.timepoints) for _ in data]
        self.n_visits_max = max(self.n_visits_per_individual) if self.n_visits_per_individual else 0  # handle case when empty dataset

        values = torch.zeros((self.n_individuals, self.n_visits_max, self.dimension))
        padding_mask = torch.zeros_like(values)

        # TODO missing values in mask ?

        for i, nb_vis in enumerate(self.n_visits_per_individual):
            # PyTorch 1.10 warns: Creating a tensor from a list of numpy.ndarrays is extremely slow.
            # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            # TODO: IndividualData.observations is really badly constructed (list of numpy 1D arrays), we should change this...
            indiv_values = torch.tensor(np.array(data[i].observations), dtype=torch.float32)
            values[i, 0:nb_vis, :] = indiv_values
            padding_mask[i, 0:nb_vis, :] = 1.

        mask_missingvalues = (~torch.isnan(values)).float()
        # mask should be 0 on visits outside individual's existing visits (he may have fewer visits than the individual with maximum nb of visits)
        # (we need to enforce it here because we padded values with 0, not with nan, so actual mask is 1 on these fictive values)
        mask = padding_mask * mask_missingvalues

        values[torch.isnan(values)] = 0.  # Set values of missing values to 0.

        self.values = values
        self.mask = mask

        # number of non-nan observations (different levels of aggregation)
        self.n_observations_per_ind_per_ft = mask.sum(dim=1).int()
        self.n_observations_per_ft = self.n_observations_per_ind_per_ft.sum(dim=0)
        self.n_observations = self.n_observations_per_ft.sum().item()

    def _construct_timepoints(self, data: Data):
        self.timepoints = torch.zeros((self.n_individuals, self.n_visits_max))
        nbs_vis = [len(_.timepoints) for _ in data]
        for i, nb_vis in enumerate(nbs_vis):
            self.timepoints[i, 0:nb_vis] = torch.tensor(data[i].timepoints)

    def _compute_L2_norm(self):
        self.L2_norm_per_ft = torch.sum(self.mask.float() * self.values * self.values, dim=(0,1)) # 1D tensor of shape (dimension,)
        self.L2_norm = self.L2_norm_per_ft.sum() # sum on all features

    def get_times_patient(self, i: int) -> torch.FloatTensor:
        """
        Get ages for patient number ``i``

        Parameters
        ----------
        i : int
            The index of the patient (<!> not its identifier)

        Returns
        -------
        :class:`torch.Tensor`, shape (n_obs_of_patient,)
            Contains float
        """
        return self.timepoints[i, :self.n_visits_per_individual[i]]

    def get_values_patient(self, i: int) -> torch.FloatTensor:
        """
        Get values for patient number ``i``, with nans.

        Parameters
        ----------
        i : int
            The index of the patient (<!> not its identifier)

        Returns
        -------
        :class:`torch.Tensor`, shape (n_obs_of_patient, dimension)
            Contains float or nans
        """
        values_with_nans = self.values[i, :self.n_visits_per_individual[i], :].clone().detach()
        nans = self.mask[i, :self.n_visits_per_individual[i], :] == 0
        values_with_nans[nans] = float('nan')
        return values_with_nans

    @staticmethod
    def _check_model_compatibility(data: Data, model: AbstractModel):
        if model.dimension is not None and data.dimension != model.dimension:
            raise LeaspyInputError(f"Unmatched dimensions: {model.dimension} (model) ≠ {data.dimension} (data).")

        if model.features is not None and data.headers != model.features:
            raise LeaspyInputError(f"Unmatched features: {model.features} (model) ≠ {data.headers} (data).")

    @staticmethod
    def _check_algo_compatibility(data: Data, algo: AbstractAlgo):
        return

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert dataset to a `DataFrame`.

        Returns
        -------
        :class:`pandas.DataFrame`
        """

        # TODO : @Raphael : On est oblige de garder une dependance pandas ? Je crois que c'est utilise juste pour l'initialisation
        # TODO : Si fait comme ca, il faut preallouer la memoire du dataframe a l'avance!
        # du modele multivarie. On peut peut-etre s'en passer?
        df = pd.DataFrame()

        for i, idx in enumerate(self.indices):
            times = self.get_times_patient(i).cpu().numpy()
            x = self.get_values_patient(i).cpu().numpy()
            df_patient = pd.DataFrame(data=x, index=times.reshape(-1), columns=self.headers)
            df_patient.index.name = 'TIME'
            df_patient['ID'] = idx
            df = pd.concat([df, df_patient])

        # df.columns = ['ID', 'TIME'] + self.headers
        df.reset_index(inplace=True)

        return df

    def move_to_device(self, device: torch.device) -> None:
        """
        Moves the dataset to the specified device.

        Parameters
        ----------
        device : torch.device
        """
        for attribute_name in dir(self):
            if attribute_name.startswith('__'):
                continue
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, torch.Tensor):
                setattr(self, attribute_name, attribute.to(device))
