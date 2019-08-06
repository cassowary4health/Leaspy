from .abstract_personalize_algo import AbstractPersonalizeAlgo
from scipy.optimize import minimize
import numpy as np
import torch

class ScipyMinimize(AbstractPersonalizeAlgo):

    def __init__(self, settings):

        # Algorithm parameters
        self.algo_parameters = settings.parameters

    def _get_individual_parameters(self,model,times,values):

        timepoints = times.reshape(1,-1,1)

        def obj(x, *args):
            ### Parameters
            model, times,values = args
            realizations = model.get_realization_object(1)
            realizations['xi'].tensor_realizations = torch.tensor(x[0],dtype=torch.float32)
            realizations['tau'].tensor_realizations = torch.tensor(x[1],dtype=torch.float32)
            realizations['sources'].tensor_realizations = torch.tensor(x[2:],dtype=torch.float32)
            xi,tau,sources = torch.tensor(x[0],dtype=torch.float32),torch.tensor(x[1],dtype=torch.float32),torch.tensor(x[2:],dtype=torch.float32)
            err = model.compute_individual_tensorized(times, (xi,tau,sources))-values
            attachement = torch.sum(err**2)
            regularity = 0
            for key,value in zip(['xi','tau','sources'],(xi,tau,sources)):
                mean = model.parameters["{0}_mean".format(key)]
                std = model.parameters["{0}_std".format(key)]
                regularity += torch.sum(model.compute_regularity_variable(value,mean,std))
            return (regularity + attachement).detach().numpy()

        res = minimize(obj,
                       x0=np.array([model.parameters["xi_mean"], model.parameters["tau_mean"]] + [0 for _ in range(
                           model.source_dimension)]),
                       args=(model,timepoints,values),
                       method="Powell"
                       )

        if res.success != True:
            print(res.success, res)

        xi, tau, sources = res.x[0], res.x[1], res.x[2:]

        return xi, tau, sources




"""
def personalize(ages, values, population_parameters):
    sqeuclidean = lambda x: np.inner(x, x)

    def obj(x, *args):
        ### Parameters
        xi, tau, sources = x[0], x[1], x[2:]
        ages, values, pop = args

        ### Likelihood : Attachment part
        noise = pop['noise']
        parallel_curve = [fit(a, np.exp(xi), tau, sources, pop) for a in ages]
        attachement = 0
        attachement = np.sum([sqeuclidean(p - values[i]) for i, p in enumerate(parallel_curve)])
        attachement /= (2.*noise)

        ### Regularity : Regularity part
        xi_mean, tau_mean = pop['xi_mean'], pop['tau_mean']
        xi_variance, tau_variance = pop['xi_variance'], pop['tau_variance']

        regularity = 0
        regularity += (tau)**2 / (2.*tau_variance)
        regularity += (xi)**2 / (2.*xi_variance)
        for i, s in enumerate(sources):
            var_source = pop['source#'+str(i)+'_variance']
            regularity += s**2 / (2.*var_source)

        return regularity + attachement

    res = minimize(obj,
                  x0=[population_parameters["xi_mean"], population_parameters["tau_mean"]] + [0 for _ in range(population_parameters["number_of_sources"])],
                  args=(ages, values, population_parameters),
                  method="Powell"
                  )

    if res.success != True:
        print(res.success, res)

    xi, tau, sources = res.x[0], res.x[1], res.x[2:]

    return xi, tau, sources

def get_individual_parameters(data, population_parameters):
    individual_parameters = {}

    for idx in np.unique(data.index.get_level_values(0)):
        #print(idx)
        indiv_df = data.loc[idx]
        indiv_df.dropna(inplace=True)
        ages = indiv_df.index.values
        observations = indiv_df.values

        xi, tau, sources = personalize(ages, observations, population_parameters)

        individual_parameters[idx] = {
            'xi': xi,
            'tau': tau,
            'sources': sources
        }

    return individual_parameters

def get_group_average_values(ages, population_parameters):

    sources = [0 for _ in range(population_parameters['number_of_sources'])]
    Y = [fit(a, 1, 0, sources, population_parameters) for a in ages]

    return ages, np.array(Y).T
"""