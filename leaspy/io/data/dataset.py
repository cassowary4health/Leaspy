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
    nb_observations_per_individuals : list[int]
        Number of observations per individual
    max_observations : int
        Maximum number of observation for one individual
    n_visits : int
        Total number of visits
    indices : list[ID]
        Order of patients

    timepoints : :class:`torch.FloatTensor`, shape (n_individuals, max_observations)
        Ages of patients at their different visits
    values : :class:`torch.FloatTensor`, shape (n_individuals, max_observations, dimension,)
        Values of patients for each visit for each feature
    mask : :class:`torch.FloatTensor`, shape (n_individuals, max_observations, dimension,)
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

        self.timepoints: torch.FloatTensor = None
        self.values: torch.FloatTensor = None
        self.mask: torch.FloatTensor = None
        self.cofactors: torch.FloatTensor = None
        self.headers = data.headers
        self.n_individuals: int = None
        self.nb_observations_per_individuals: List[int] = None
        self.max_observations: int = None
        self.dimension: int = None
        self.n_visits: int = None
        #self.individual_parameters = None
        self.indices = list(data.individuals.keys())
        self.L2_norm_per_ft: torch.FloatTensor = None # 1D float tensor, shape (dimension,)
        self.L2_norm: torch.FloatTensor = None # scalar float tensor

        if model is not None:
            self._check_model_compatibility(data, model)
        if algo is not None:
            self._check_algo_compatibility(data, algo)

        self._construct_values(data)
        self._construct_timepoints(data)
        self._construct_cofactors(data)
        self._compute_L2_norm()

    def _construct_values(self, data: Data):

        batch_size = data.n_individuals
        x_len = [len(_.timepoints) for _ in data]
        channels = data.dimension
        values = torch.zeros((batch_size, max(x_len), channels), dtype=torch.float32)
        padding_mask = torch.zeros((batch_size, max(x_len), channels), dtype=torch.float32)

        # TODO missing values in mask ?

        for i, d in enumerate(x_len):
            # PyTorch 1.10 warns: Creating a tensor from a list of numpy.ndarrays is extremely slow.
            # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            # TODO: IndividualData.observations is really badly constructed (list of numpy 1D arrays), we should change this...
            indiv_values = torch.tensor(np.array(data[i].observations), dtype=torch.float32)
            values[i, 0:d, :] = indiv_values
            padding_mask[i, 0:d, :] = 1.

        mask_missingvalues = (~torch.isnan(values)).float()
        # mask should be 0 on visits outside individual's existing visits (he may have fewer visits than the individual with maximum nb of visits)
        # (we need to enforce it here because we padded values with 0, not with nan, so actual mask is 1 on these fictive values)
        mask = padding_mask * mask_missingvalues

        values[torch.isnan(values)] = 0.  # Set values of missing values to 0.

        self.n_individuals = batch_size
        self.max_observations = max(x_len)
        self.nb_observations_per_individuals = x_len # list of length n_individuals
        self.dimension = channels
        self.values = values
        self.mask = mask
        self.n_visits = data.n_visits

        # number of non-nan observations (different levels of aggregation)
        self.n_observations_per_ind_per_ft = mask.sum(dim=1).int() # 2D int tensor of shape(n_individuals,dimension)
        self.n_observations_per_ft = self.n_observations_per_ind_per_ft.sum(dim=0) # 1D int tensor of shape(dimension,)
        self.n_observations = self.n_observations_per_ft.sum().item() # scalar (int)

    def _construct_timepoints(self, data: Data):
        self.timepoints = torch.zeros([self.n_individuals, self.max_observations], dtype=torch.float32)
        x_len = [len(_.timepoints) for _ in data]
        for i, d in enumerate(x_len):
            self.timepoints[i, 0:d] = torch.tensor(data[i].timepoints, dtype=torch.float32)

    def _compute_L2_norm(self):
        self.L2_norm_per_ft = torch.sum(self.mask.float() * self.values * self.values, dim=(0,1)) # 1D tensor of shape (dimension,)
        self.L2_norm = self.L2_norm_per_ft.sum() # sum on all features

    def _construct_cofactors(self, data: Data):
        self.cofactors_dimension = len(data.cofactors)

        if self.cofactors_dimension > 0:
            self.cofactors = torch.zeros((data.n_individuals, self.cofactors_dimension), dtype=torch.float32)

            # dictionnary used to map categorical cofactors to numerical values
            # e.g. {"Female", "Male"} becomes {0.0, 1.0}
            self.cofactors_association_dict = {}

            def _categorize(cofactor_name: str, value: Any):
                if cofactor_name not in self.cofactors_association_dict:
                    self.cofactors_association_dict[cofactor_name] = {"count": 0.0}

                if isinstance(value, str):
                    if value not in self.cofactors_association_dict[cofactor_name]:
                        self.cofactors_association_dict[cofactor_name][value] = self.cofactors_association_dict[cofactor_name]["count"]
                        self.cofactors_association_dict[cofactor_name]["count"] += 1.0

                    return self.cofactors_association_dict[cofactor_name][value]

                else:
                    return value

            for i, subject in data.iter_to_idx.items():
                subject_cofactors = data.get_by_idx(subject).cofactors
                for j, (cofactor_name, value) in enumerate(subject_cofactors.items()):
                    self.cofactors[i,j] = _categorize(cofactor_name, value)
                        

    def get_times_patient(self, i: int) -> torch.FloatTensor:
        """
        Get ages for patient number ``i``

        Returns
        -------
        :class:`torch.Tensor`, shape (n_obs_of_patient,)
            Contains float
        """
        return self.timepoints[i, :self.nb_observations_per_individuals[i]]

    def get_values_patient(self, i: int) -> torch.FloatTensor:
        """
        Get values for patient number ``i``

        Returns
        -------
        :class:`torch.Tensor`, shape (n_obs_of_patient, dimension)
            Contains float or nans
        """
        values = self.values[i, :self.nb_observations_per_individuals[i], :]
        # mask = self.mask[i].clone().cpu().detach().numpy()[:values.shape[0],:]
        mask = self.mask[i].clone().cpu().detach()[:values.shape[0], :]
        # mask[mask==0] = np.nan
        mask[mask == 0] = float('NaN')
        values_with_na = values * mask
        return values_with_na

    def get_cofactors_patient(self, i: int) -> torch.FloatTensor:
        if hasattr(self, "cofactors"):
            return self.cofactors[i,:].unsqueeze(-1)
        else:
            raise RuntimeError("The Dataset does not have any cofactor data")

    @staticmethod
    def _check_model_compatibility(data: Data, model: AbstractModel):
        if model.dimension is None:
            return
        if data.dimension != model.dimension:
            raise LeaspyInputError(f"Unmatched dimensions: {model.dimension} (model) ≠ {data.dimension} (data).")

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
        """

        # I don't know if we should make the moved attributes explicit...
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, torch.Tensor):
                setattr(self, attribute_name, attribute.to(device))
