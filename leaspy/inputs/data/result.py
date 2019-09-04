import numpy as np


class Result:
    """
    Used as output from personalize
    """
    def __init__(self, data, individual_parameters, noise_std=None):
        self.data = data
        self.individual_parameters = individual_parameters
        self.noise_std = noise_std

    #def load_covariables(self, covariables, csv):
    #   self.covariables = covariables

    def get_parameter_distribution(self, parameter, cofactor=None):
        # If there is no cofactor to take into account
        parameter_distribution = self.individual_parameters[parameter].detach().numpy().tolist()
        # TODO : parameter_distribution is of size (N_subjects, N_dimension_of_parameter)
        # TODO : This leads to a problem in term of shape, so the quick fix is the np.ravel.
        # TODO : This will create an unexpected behavior for sources
        parameter_distribution = np.ravel(parameter_distribution)


        if cofactor is None:
            return parameter_distribution

        # If the distribution as asked for different cofactor values
        possibilities = np.unique([_.cofactors[cofactor] for _ in self.data if _.cofactors[cofactor] is not None])
        distributions = {}
        for p in possibilities:
            if p not in distributions.keys():
                distributions[p] = []

            for i, v in enumerate(parameter_distribution):
                if self.data[i].cofactors[cofactor] == p:
                    distributions[p].append(v)
        return distributions


    def get_cofactor_distribution(self, cofactor):
        return [d.cofactors[cofactor] for d in self.data]

    def get_patient_individual_parameters(self, idx):

        indices = list(self.data.individuals.keys())
        idx_number = int(np.where(np.array(indices) == idx)[0]) # TODO save somewhere the correspondance : in data probably ???

        patient_dict = dict.fromkeys(self.individual_parameters.keys())

        for variable_ind in list(self.individual_parameters.keys()):
            patient_dict[variable_ind] = self.individual_parameters[variable_ind][idx_number]

        return patient_dict

