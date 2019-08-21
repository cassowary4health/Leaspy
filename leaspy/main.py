import pandas as pd
import json
import torch
import numpy as np
import scipy.stats as st
from torch.distributions import normal

from leaspy.inputs.data.dataset import Dataset
from leaspy.inputs.settings.model_settings import ModelSettings
from leaspy.models.model_factory import ModelFactory
from leaspy.algo.algo_factory import AlgoFactory
from leaspy.utils.output.visualization.plotter import Plotter


class Leaspy:
    def __init__(self, model_name):
        self.type = model_name
        self.model = ModelFactory.model(model_name)
        self.plotter = Plotter()

    @classmethod
    def load(cls, path_to_model_settings):
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.is_initialized = True
        return leaspy

    def save(self, path):
        self.model.save(path)

    def save_individual_parameters(self, path,individual_parameters):
        for key1 in individual_parameters.keys():
            for key2 in   individual_parameters[key1]:
                if type(individual_parameters[key1][key2]) not in [list]:
                    individual_parameters[key1][key2]= individual_parameters[key1][key2].tolist()

        with open(path, 'w') as fp:
            json.dump(individual_parameters, fp)

    def load_individual_parameters(self, path):
        with open(path, 'r') as f:
            individual_parameters = json.load(f)

        return individual_parameters

    def to_dataset(self,data):
        dataset = Dataset(data,model=self.model)
        return dataset


    def fit(self, data, algorithm_settings):

        algorithm = AlgoFactory.algo(algorithm_settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        if not self.model.is_initialized:
            self.model.initialize(dataset)
        algorithm.run(dataset, self.model)

    def personalize(self, data, settings):

        print("Load personalize algorithm")
        algorithm = AlgoFactory.algo(settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)

        # Predict
        print("Launch predict algo")
        individual_parameters = algorithm.run(self.model,dataset)

        #TODO and algorithm.personalize output, with the distributions ???

        return individual_parameters





    def simulate(self,dataset,model,param_ind,N_indiv,noise_scale=1.,ref_age=79,interval=1.):
        # simulate according to same distrib as param_ind
        param = []
        for a in param_ind:
            for i in range(a.shape[1]):
                param.append(a[:, i].detach().numpy())
        param = np.array(param)

        kernel = st.gaussian_kde(param)

        # Get metrics from Data
        n_points = np.mean(dataset.nb_observations_per_individuals)
        data_sim = pd.DataFrame(columns=['ID','TIME']+dataset.headers)
        indiv_param = {}

        noise = normal.Normal(0, noise_scale*self.model.parameters['noise_std'])
        t0 = self.model.parameters['tau_mean'].detach().numpy()
        v0 = self.model.parameters['xi_mean'].detach().numpy()
        for idx in range(N_indiv):
            this_indiv_param ={}
            this_indiv_param[idx] ={}
            sim = kernel.resample(1)[:,0]
            this_indiv_param[idx]['xi'] = sim[0]
            this_indiv_param[idx]['tau'] = sim[1]
            if model.name!="univariate":
                this_indiv_param[idx]['sources'] = sim[2:]
            indiv_param.update(this_indiv_param)
            age_diag = (ref_age - t0) *np.exp(v0)/ np.exp(sim[0]) + sim[1]
            # Draw the number of visits
            n_visits = np.random.randint(max(2,n_points-3),dataset.max_observations)
            timepoints = np.linspace(age_diag-n_visits*interval/2,age_diag+n_visits*interval/2,n_visits)
            timepoints = torch.tensor(timepoints,dtype=torch.float32).unsqueeze(0)

            values = self.model.compute_individual_tensorized(timepoints,self.model.param_ind_from_dict(this_indiv_param))
            values = values + noise_scale*noise.sample(values.shape)
            values = torch.clamp(values,0,1).squeeze(0)

            ten_idx = torch.tensor([idx] * (n_visits),dtype = torch.float32)
            val = torch.cat([ten_idx.unsqueeze(1), timepoints.squeeze(0).unsqueeze(1), values], dim=1)
            truc = pd.DataFrame(val.detach().numpy(), columns=['ID','TIME']+dataset.headers)
            data_sim = data_sim.append(truc)

        return data_sim, indiv_param
