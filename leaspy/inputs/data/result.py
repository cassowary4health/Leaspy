
class Result:
    """
    Used as output from personalize
    """
    def __init__(self, data, individual_parameters):
        self.data = data
        self.individual_parameters = individual_parameters
        self.covariables = None

    def set_covariables(self, covariables):
        self.covariables = covariables


