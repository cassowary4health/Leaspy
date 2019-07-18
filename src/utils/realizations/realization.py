import torch

class Realization:
    def __init__(self, name, shape, variable_type):
        self.name = name
        self.shape = shape
        self.variable_type = variable_type
        self._tensor_realizations = None

    def initialize(self, data, model):
        print("Initialize realizations of {0}".format(self.name))
        if self.variable_type == "population":
            self._tensor_realizations = torch.Tensor([model.parameters[self.name]]).reshape(self.shape)

        elif self.variable_type == 'individual':

            distribution = torch.distributions.normal.Normal(loc=model.parameters["{0}_mean".format(self.name)],
                                                             scale=model.parameters["{0}_std".format(self.name)])
            self._tensor_realizations = distribution.sample(sample_shape = (data.n_individuals, self.shape[0], self.shape[1]))
        else:
            raise ValueError("Variable Type Not Known")

    @property
    def tensor_realizations(self):
        return self._tensor_realizations

    @tensor_realizations.setter
    def tensor_realizations(self, tensor_realizations):
        self._tensor_realizations = tensor_realizations

    def set_tensor_realizations_element(self, element, dim):
        self._tensor_realizations[dim] = element


    def __str__(self):
        str = "Realization of {0} \n".format(self.name)
        str += "Shape : {0} \n".format(self.shape)
        str += "Variable type : {0} \n".format(self.variable_type)
        return str
