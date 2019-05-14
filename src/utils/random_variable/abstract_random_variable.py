

class AbstractRandomVariable:
    def __init__(self):
        pass

    def compute_loglikelihood(self, x):
        raise NotImplementedError