import numpy as np
import torch

class AbstractSampler():

    def __init__(self,info,n_patients):
        self.name = info["name"]
        self.temp_length = 25
        if info["type"] == "population":
            self.type = 'pop'
        elif info["type"] == "individual":
            self.type = 'ind'
            self.temp_length *= n_patients
        self.acceptation_temp = [0.0] * self.temp_length
        return

    def _group_metropolis_step(self, alpha):
        accepted = torch.tensor(1. * (torch.rand(alpha.size(0)) < alpha), dtype=torch.float32)
        return accepted

    def _metropolis_step(self, alpha):
        accepted = 0
        if alpha >= 1:
            # Case 1: we improved the LogL
            accepted = 1
        else:
            # Case 2: we decreased the LogL
            # Sample a realization from uniform law
            realization = np.random.uniform(low=0, high=1)
            # Choose to keep a lesser parameter value from it
            if realization < alpha:
                accepted = 1
        return accepted

    def _update_acceptation_rate(self, accepted):
        self.acceptation_temp.extend(accepted)
        self.acceptation_temp = self.acceptation_temp[len(accepted):]