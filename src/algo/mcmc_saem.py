import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algo_reader import AlgoReader
from src import default_algo_dir
from src.utils.sampler import Sampler
import matplotlib.pyplot as plt

import numpy as np

class MCMCSAEM(AbstractAlgo):

    def __init__(self):
        data_dir = os.path.join(default_algo_dir, "default_mcmc_saem_parameters.json")
        reader = AlgoReader(data_dir)

        if reader.algo_type != 'mcmc_saem':
            raise ValueError("The default mcmc saem parameters are not of random_sampling type")


        self.realizations = None
        self.task = None
        self.algo_parameters = reader.parameters

        self.samplers_pop = None
        self.samplers_ind = None

        self.path_output = "output/"

    def run(self, data, model, seed=None, path_output=None):

        self._initialize_seed(seed)
        self._initialize_path_output(path_output)

        realizations = model.initialize_realizations(data)
        self.initialize_samplers(model)




        iters = []
        noise_var_list = []
        xi_mean_list = []
        xi_std_list = []
        tau_mean_list = []
        tau_std_list = []
        p0_list = []

        mu_list = []
        intercept_var_list = []


        for iteration in range(self.algo_parameters['n_iter']):
            self.iter(data, model, realizations)

            if iteration % 100 == 0:

                """
                model.plot(data, iteration, realizations, self.path_output)
                """

                reals_pop, reals_ind = realizations

                """

                fig, ax = plt.subplots(6, 1, figsize=(6,12))
                
                iters.append(iteration)
                noise_var_list.append(model.model_parameters['noise_var'])

                xi_mean_list.append(np.mean([x.detach().numpy() for _,x in reals_ind['xi'].items()]))
                xi_std_list.append(np.std([x.detach().numpy() for _,x in reals_ind['xi'].items()]))
                tau_mean_list.append(np.mean([x.detach().numpy() for _,x in reals_ind['tau'].items()]))
                tau_std_list.append(np.std([x.detach().numpy() for _,x in reals_ind['tau'].items()]))
                p0_list.append((reals_pop['p0'].detach().numpy()))

                ax[0].plot(iters, noise_var_list)
                ax[0].set_title('Noise variance')
                ax[1].plot(iters, xi_mean_list)
                ax[1].set_title('Xi mean, rate {0}'.format(np.mean(self.samplers_ind['xi'].acceptation_temp)))
                ax[2].plot(iters, xi_std_list)
                ax[2].set_title('Xi std')
                ax[3].plot(iters, tau_mean_list)
                ax[3].set_title('tau mean, rate {0}'.format(np.mean(self.samplers_ind['tau'].acceptation_temp)))
                ax[4].plot(iters, tau_std_list)
                ax[4].set_title('Tau std')
                ax[5].plot(iters, p0_list)
                ax[5].set_title('p0, rate {0}'.format(np.mean(self.samplers_pop['p0'].acceptation_temp)))


                
                intercept_var_list.append(np.var([x.detach().numpy() for _,x in reals_ind['intercept'].items()]))
                mu_list.append(np.mean([x.detach().numpy() for _,x in reals_ind['intercept'].items()]))

                ax[0].plot(iters, noise_var_list)
                ax[0].set_title('Noise')
                ax[1].plot(iters, intercept_var_list)
                ax[1].set_title('Intercept var list')
                ax[2].plot(iters, mu_list)
                ax[2].set_title("Mu list")
                

                plt.tight_layout()
                plt.savefig(os.path.join(output_path,'Convergence_parameters.pdf'))
                """

                print("=============================================")
                print("ITER ---- {0}".format(iteration))
                print(model)
                print("       Samplers      ")

                # Samplers
                for variable_ind in model.reals_ind_name:
                    print(self.samplers_ind[variable_ind])


    def iter(self, data, model, realizations):

        reals_pop, reals_ind = realizations

        # Sample step
        for key in reals_pop.keys():

            # Old loss
            previous_reals_pop = reals_pop[key]
            previous_attachment = model.compute_attachment(data, reals_pop, reals_ind)
            previous_regularity = model.compute_regularity(data, reals_pop, reals_ind)
            previous_loss = previous_attachment + previous_regularity
            # New loss
            reals_pop[key] = reals_pop[key] + self.samplers_pop[key].sample()
            new_attachment = model.compute_attachment(data, reals_pop, reals_ind)
            new_regularity = model.compute_regularity(data, reals_pop, reals_ind)
            new_loss = new_attachment + new_regularity

            alpha = np.exp(-(new_loss-previous_loss).detach().numpy())

            # Compute acceptation
            accepted = self.samplers_pop[key].acceptation(alpha)

            # Revert if not accepted
            if not accepted:
                reals_pop[key] = previous_reals_pop


        for idx in reals_ind.keys():
            for key in reals_ind[idx].keys():

                # Save previous realization
                previous_reals_ind = reals_ind[idx][key]
                #print(previous_reals_ind)

                # Compute previous loss

                previous_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])
                previous_individual_regularity = model.compute_individual_regularity(reals_ind[idx])
                previous_individual_loss = previous_individual_attachment + previous_individual_regularity

                # Sample a new realization
                reals_ind[idx][key] = reals_ind[idx][key] + self.samplers_ind[key].sample()

                # Compute new loss
                new_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])
                new_individual_regularity = model.compute_individual_regularity(reals_ind[idx])
                new_individual_loss = new_individual_attachment + new_individual_regularity
                
                alpha = np.exp(-(new_individual_loss - previous_individual_loss).detach().numpy())
                #print(alpha)
                #print(key)
                #print('--')


                #print("New loss {0} : Previous Loss : ".format(new_individual_loss-previous_individual_loss))
                #print("New att {0} : Previous att : ".format(new_individual_attachment-previous_individual_attachment))
                #print("New reg {0} : Previous reg : ".format(new_individual_regularity-previous_individual_regularity))
                #print("alpha : {0}".format(alpha))
                """
                previous_attachment = model.compute_attachment(data, reals_pop, reals_ind)
                previous_regularity = model.compute_regularity(data, reals_pop,reals_ind)
                previous_loss = previous_attachment + previous_regularity

                # Sample a new realization
                reals_ind[key][idx] = reals_ind[key][idx] + self.samplers_ind[key].sample()

                new_attachment = model.compute_attachment(data, reals_pop, reals_ind)
                new_regularity = model.compute_regularity(data, reals_pop, reals_ind)
                new_loss = new_attachment + new_regularity

                alpha = np.exp(-(new_loss - previous_loss).detach().numpy())"""

                # Compute acceptation
                accepted = self.samplers_ind[key].acceptation(alpha)

                # Revert if not accepted
                if not accepted:
                    reals_ind[idx][key] = previous_reals_ind



        # Maximization step
        if self.algo_parameters['estimate_population_parameters']:
            model.update_sufficient_statistics(data, reals_ind, reals_pop)

        self.realizations = realizations

        #print("Previous loss {0}".format(previous_individual_loss))
        #print("Previous individual regularity{0}".format(previous_individual_regularity))
        #print("Previous individual attachment{0}".format(previous_individual_attachment))


    def get_realizations(self):
        return self.realizations

    def set_mode(self, task):
        self.task = task
        if self.task == 'fit':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = True
        elif self.task == 'predict':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = False


    def initialize_samplers(self, model):

        pop_name = model.reals_pop_name
        ind_name = model.reals_ind_name

        self.samplers_pop = dict.fromkeys(pop_name)
        self.samplers_ind = dict.fromkeys(ind_name)

        # TODO Change this arbitrary parameters --> samplers parameters ???
        for key in pop_name:
            self.samplers_pop[key] = Sampler(key, 0.01, 50)

        for key in ind_name:
            self.samplers_ind[key] = Sampler(key, np.sqrt(model.model_parameters["{0}_var".format(key)])/2, 200)
