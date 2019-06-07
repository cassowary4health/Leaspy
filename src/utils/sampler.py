import numpy as np


# TODO Numba this
class Sampler:
    def __init__(self, name, std, temp_length=100):
        self.name = name
        self.std = std

        # Acceptation rate
        self.temp_length = temp_length
        self.acceptation_temp = []
        self.counter_acceptation = 0

    def sample(self):
        return np.random.normal(loc=0, scale=self.std)

    # TODO Numba this
    def acceptation(self, alpha):
        """
        Update according to the difference in negative log likelihood: nll_new - nll_old
        :param alpha:
        :return:
        """

        alpha = min(alpha, 1)

        accepted = 1

        # If new loss > previous loss
        if alpha < 1:

            # Sample a realization from uniform law
            realization = np.random.uniform(low=0, high=1)

            # Choose to keep a lesser parameter value from it
            if realization > alpha:
                accepted = 0

        # Update acceptance rate of sampler
        self.update_acceptation_rate(accepted)
        return accepted


    def update_acceptation_rate(self, accepted):

        self.acceptation_temp.append(accepted)
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
            self.reset_acceptation_temp()

    def reset_acceptation_temp(self):
        self.acceptation_temp = []
        self.counter_acceptation = 0

    def __str__(self):
        return "Sampler {0}, std:{1} , rate:{2}".format(self.name, self.std, np.mean(self.acceptation_temp))





