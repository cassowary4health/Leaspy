from src.inputs.individual_data import IndividualData


class Data():
    def __init__(self):
        self.indices = []
        self.individuals = []

    def add_individual(self, individual):
        if individual.idx in self.indices:
            raise ValueError("There already")

        self.indices.append(individual.idx)
        self.individuals.append(individual)
