import torch
import numpy as np
from bisect import bisect

class IndividualData():
    def __init__(self, idx):
        self.idx = idx
        self.individual_parameters = None
        self.timepoints = None
        self.raw_observations = None
        self.tensor_observations = None

        # Metrics
        self.n_visits = 0
        self.n_observations = 0


    def add_observation(self, timepoint, values):
        if self.timepoints is None:
            self.timepoints = []
            self.raw_observations = []

        if timepoint in self.timepoints:
            raise ValueError('You are trying to overwrite the observation of a subject')

        index = bisect(self.timepoints, timepoint)
        self.timepoints.insert(index, timepoint)
        self.raw_observations.insert(index, values)

        # Torch
        self.tensor_observations = torch.from_numpy(np.array(self.raw_observations)).float()
        self.tensor_timepoints = torch.from_numpy(np.array(self.timepoints)).float()


        # Update metrics
        self.n_visits += 1
        self.n_observations += np.array(values).shape[0]
