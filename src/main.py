from src.inputs.data.dataset import Dataset
from src.inputs.model_settings import ModelSettings
from src.inputs.data.data import Data

from src.models.model_factory import ModelFactory

from src.algo.algo_factory import AlgoFactory
from src.utils.output_manager import OutputManager
import json
import torch
import numpy as np

from src.utils.realizations.collection_realization import CollectionRealization
from src.utils.realizations.realization import Realization

class Leaspy:
    def __init__(self, model_name):
        self.type = model_name
        self.model = ModelFactory.model(model_name)

    @classmethod
    def load(cls, path_to_model_settings):
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        leaspy.model.is_initialized = True
        return leaspy

    def save(self, path):
        self.model.save(path)

    def fit(self, data, algorithm_settings):

        algorithm = AlgoFactory.algo(algorithm_settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        if not self.model.is_initialized:
            self.model.initialize(dataset)
        algorithm.run(dataset, self.model)

    def personalize(self, data, prediction_settings):

        print("Load predict algorithm")
        algorithm = AlgoFactory.algo(prediction_settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)

        # Predict
        print("Launch predict algo")
        realizations = algorithm.run(dataset, self.model)

        # Individual attachment
        noise = (self.model.compute_sum_squared_tensorized(dataset, realizations).sum()/(dataset.n_visits*dataset.dimension)).detach().numpy().tolist()
        print("Noise : {0}".format(noise))

        #TODO and algorithm.personalize output, with the distributions ???

        # Keep the individual variables
        data.realizations = realizations

        return data



    def simulate(self, data, n_individuals):

        print("WARNING : simulate is yet to be improved")

        # Get metrics from Data
        patient_timepoint_number = []
        patient_baseline_age = []
        duration_between_visits = []

        # Get metrics from real dataset
        for key, individual in data.individuals.items():
            patient_timepoint_number.append(len(individual.timepoints))
            patient_baseline_age.append(np.min(individual.timepoints))
            duration_between_visits.append(np.mean(np.diff(individual.timepoints)).tolist())

        # Generate timepoints patients
        new_dataset_infos = {}

        for i in range(n_individuals):
            # Instanciate the patient in the dictionnary
            new_dataset_infos[i] = {"id":i,
                                    "timepoints":[],
                                    "n_visits": None}
            # Draw the number of visits
            n_visits = np.random.choice(patient_timepoint_number, 1)
            new_dataset_infos[i]["n_visits"] = int(n_visits[0])
            # Draw a beseline age
            current_age = np.random.choice(patient_baseline_age, 1)
            # For all visits
            for visit_num in range(int(n_visits)):
                # Draw a duration to next visit and update the current age
                new_dataset_infos[i]['timepoints'].append(float(current_age))
                duration = float(np.random.choice(duration_between_visits, 1))
                current_age += duration

        # Compute tensor of timepoints and mask
        max_observations = max([new_dataset_infos[i]["n_visits"] for i in range(n_individuals)])
        timepoints_tensor = torch.zeros([n_individuals, max_observations])

        mask_tensor = torch.zeros([n_individuals, max_observations, self.model.dimension])

        for i, infos in new_dataset_infos.items():
            timepoints_tensor[i, 0:infos["n_visits"]] = torch.Tensor(infos['timepoints'])

            mask_tensor[i, 0:infos["n_visits"],:] = 1.0


        ## Compute models from the realizations + model parameters
        # Instanciate realizations
        realizations = self.model.get_realization_object(n_individuals)
        # TODO, draw better realizations than these ones
        for key, value in self.model.random_variable_informations().items():
            if value["type"] == "individual":
                realizations.reals_ind_variable_names.append(key)
                realizations.realizations[key] = Realization(key, value["shape"], value["type"])
                realizations.realizations[key].initialize(n_individuals, self.model, scale_individual=1.0)

        # Create dummy Data object : timepoints but no values (ar at nans ?)
        class DummyDataset():
            def __init__(self, timepoints, mask):
                self.timepoints = timepoints
                self.mask = mask

        dummy_dataset = DummyDataset(timepoints_tensor, mask_tensor)

        # Compute model
        model_values = self.model.compute_individual_tensorized(dummy_dataset, realizations)

        # Add the noise + constraints (if sigmoid then limited between 0-1 for example)
        normal_distr = torch.distributions.normal.Normal(loc=0, scale=self.model.parameters['noise_std'])
        model_values = model_values + normal_distr.sample(sample_shape=model_values.shape)
        # TODO add constraints

        # Create synthetic Dataset
        indices = list(range(n_individuals))
        timepoints = [new_dataset_infos[i]['timepoints'] for i in range(n_individuals)]
        values = [model_values[i][:new_dataset_infos[i]["n_visits"],:].detach().numpy() for i in range(n_individuals)]
        simulated_data = Data.from_individuals(indices, timepoints, values, data.headers)



        # Add the individual parameters
        for i, idx in enumerate(indices):
            for key, value in self.model.random_variable_informations().items():
                if value["type"] == "individual":
                    simulated_data.individuals[idx].add_individual_parameters(key, realizations[key].tensor_realizations[i].detach().numpy())

        # TODO pas ouf
        simulated_data.realizations = realizations

        return simulated_data








