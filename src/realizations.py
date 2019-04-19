

class Realizations():
    """
    Realizations
    Population parameters that need to be considered as realizations
    Individual parameters
    """

    def __init__(self):
        self.realizations_ind = {}
        self.realizations_pop = {}

    def set_realizations_ind(self, reals_ind):
        self.realizations_ind = reals_ind

    def set_realizations_pop(self, reals_pop):
        self.realizations_pop = reals_pop
