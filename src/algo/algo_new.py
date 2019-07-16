

class AlgorithmNew:
    # TODO : Create an abstract model class
    def __init__(self, parameters):
        self.current_iteration = 0
        self.number_of_iterations = parameters['number_of_iterations']


    def run(self, dataset, model):
        self.initialize(model, dataset)

        # Iterations
            # MCMC sampling
            # Statistiques suffisantes
            # Maximization

    def initialize(self, model, dataset):
        model.initialize_MCMC_toolbox(dataset)
        realizations = model.initialize_realizations

    def update(self):
        return 0

