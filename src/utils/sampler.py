import numpy as np


# TODO Numba this
class Sampler:
    def __init__(self, info, n_patients, temp_length=None):


        self.name = info["name"]

        self.std = None
        self.temp_length = 25

        if info["type"] == "population":
            self.std = 0.005
        elif info["type"] == "individual":
            self.std = 0.1
            self.temp_length *= n_patients

        if temp_length is not None:
            self.temp_length = temp_length

        # For sampling options
        self.shape = info["shape"]
        self.rv_type = info["rv_type"]

        self.samplewithshape = False
        if self.rv_type == "gaussian" and self.shape != (1,1):
            self.samplewithshape = True

        # Acceptation rate
        self.acceptation_temp = [0.0] * self.temp_length
        self.counter_acceptation = 0


    def sample(self):
            return self.sample_withoutshape()



    def sample_withoutshape(self):
        return np.random.normal(loc=0, scale=self.std)

    def sample_withshape(self):
        return np.random.normal(loc=0, scale=self.std, size=self.shape)

    # TODO Numba this
    def acceptation(self, alpha):
        """
        Update according to the difference in negative log likelihood: nll_new - nll_old
        :param alpha:
        :return:
        """

        accepted = 0

        if alpha >= 1:
            # Case 1: we improved the LogL
            accepted = 1

        else:
            # Case 2: we decreased the LogL

            # Sample a realization from uniform law
            realization = np.random.uniform(low=0, high=1)

            # Choose to keep a lesser parameter value from it
            if realization < alpha:
                accepted = 1

        # Update acceptance rate of sampler
        self.update_acceptation_rate(accepted)
        return accepted


    def update_acceptation_rate(self, accepted):

        self.acceptation_temp.insert(0, accepted)
        self.acceptation_temp.pop()

        self.counter_acceptation += 1

        if self.counter_acceptation == self.temp_length:
            #print("Rate for sampler {0} : {1}".format(self.name, np.mean(self.acceptation_temp)))



            # Update the std of sampling so that expected rate is reached

            if np.mean(self.acceptation_temp) < 0.2:
                self.std = 0.9 * self.std
                #print("Decreased std of sampler-{0}".format(self.name))
            elif np.mean(self.acceptation_temp) > 0.4:
                self.std = 1.1 * self.std
                #print("Increased std of sampler-{0}".format(self.name))


            # reset acceptation temp list
            self.counter_acceptation = 0



    def __str__(self):
        return "Sampler {0}, std:{1} , rate:{2}".format(self.name, self.std, np.mean(self.acceptation_temp))





