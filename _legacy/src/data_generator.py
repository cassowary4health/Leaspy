import numpy as np
#from tests import test_data_dir
from src.inputs.data import Data
from src.inputs.data.individual_data import IndividualData
import torch

#test_data_dir = os.path.join(os.path.dirname(__file__), "_data")
#test_data_dir = "tests/_data/"
#path_to_model_parameters = os.path.join(test_data_dir, '_generate_data', 'model_settings_univariate.json')
#leaspy = Leaspy.from_model_settings(path_to_model_parameters)

def generate_data_from_model(model, n_patients = 10, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_visits_per_patients = 6

    # Initialize dummy data
    dummy_data = Data()
    dummy_data.dimension = model.dimension

    # Add individual idx
    for idx in range(n_patients):
        # Create the individual with dummy values
        individual = IndividualData(idx=idx)
        mean_timepoint = np.random.normal(loc=0, scale=1)
        timepoints = mean_timepoint + np.linspace(-0.3, 0.3, 6)
        for timepoint in timepoints:
            individual.add_observation(timepoint=[timepoint], values=[-1]*model.dimension)

        # Add individual to the data
        dummy_data.add_individual(individual)

    # Generate realizations for these patients
    reals_pop, reals_ind = model.get_realization_object(dummy_data)

    for idx in dummy_data.indices:
        # Compute model for these patients
        values_patient = model.compute_individual(individual=dummy_data[idx], reals_pop=reals_pop, real_ind=reals_ind[idx])
        values_patient += torch.Tensor(np.random.normal(loc=0, scale=np.sqrt(model.model_parameters['noise_var']), size=values_patient.shape))

        values_patient[values_patient > 1] = 1
        values_patient[values_patient < 0] = 0

        # Set the observations
        dummy_data[idx].tensor_observations = values_patient

    return dummy_data