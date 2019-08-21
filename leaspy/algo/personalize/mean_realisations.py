from .abstract_personalize_algo import AbstractPersonalizeAlgo
from scipy.optimize import minimize
import numpy as np
import torch
import csv
import pandas as pd
import os

class MeanReal(AbstractPersonalizeAlgo):

    def __init__(self, settings):

        # Algorithm parameters
        super().__init__(settings)

    def run(self,model,data):
        print(self.algo_parameters)
        path_to_parameters_convergence = os.path.join(self.outputs_path,"parameter_convergence/")
        df_xi = pd.read_csv(path_to_parameters_convergence + "xi.csv", header=None, index_col=0)
        xi_mean = df_xi.mean()
        df_tau = pd.read_csv(path_to_parameters_convergence + "tau.csv", header=None, index_col=0)
        tau_mean = df_tau.mean()
        individual_parameters = {}
        for idx in range(len(xi_mean)):
            sources_id = []
            for i in range(model.source_dimension):
                sources_temp = []
                with open(path_to_parameters_convergence + 'sources'+str(i)+'.csv') as csvDataFile:
                    csvReader = csv.reader(csvDataFile)
                    for row in csvReader:
                        sources_temp.append(float(row[idx+1]))
                sources_id.append(np.mean(sources_temp))

            individual_parameters[data.indices[idx]] = {
                'xi': xi_mean[idx+1],
                'tau': tau_mean[idx+1],
                'sources': sources_id
            }

        return individual_parameters




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