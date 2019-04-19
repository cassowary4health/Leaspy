from src.inputs.individual_data import IndividualData


class Data():
    def __init__(self):
        self.indices = []
        self.individuals = []

        # Metrics
        self.n_individuals = 0
        self.n_visits = 0
        self.n_observations = 0

    def add_individual(self, individual):
        if individual.idx in self.indices:
            raise ValueError("There already")

        self.indices.append(individual.idx)
        self.individuals.append(individual)

        self.update_metrics(individual)

    def update_metrics(self, individual):
        self.n_individuals += 1
        self.n_visits += individual.n_visits
        self.n_observations += individual.n_observations

    def __getitem__(self, index):
         return self.individuals[index]

    def __iter__(self):
        return iter(self.individuals)