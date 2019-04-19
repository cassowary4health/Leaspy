import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algo_reader import AlgoReader
from src import default_algo_dir


class GreedySampling(AbstractAlgo):

    def __init__(self):
        data_dir = os.path.join(default_algo_dir, "default_greedy_sampling_parameters.json")
        reader = AlgoReader(data_dir)

        if reader.algo_type != 'greedy_sampling':
            raise ValueError("The default greedy sampling parameters are not of greedy_sampling type")

        self.parameters = reader.parameters

    def run(self, data, model):
        for iter in range(self.parameters['n_iter']):
            self.iter(data, model)

    def iter(self, data, model):
        # Compute loss
        attachment = model.compute_attachment(data)
        regularity = 0
        loss = attachment + regularity

        # Population parameters
        model.model_parameters

        # Individual random effects




        # Do backward
        loss.backward()
        with torch.no_grad():
            for key in model.model_parameters:
                # TODO not model specific optimization of parameters : sigma ? When grad is none
                if model.model_parameters[key].grad is not None:
                    model.model_parameters[key] -= self.parameters['learning_rate'] * model.model_parameters[key].grad
                    model.model_parameters[key].grad.zero_()
        return 0



