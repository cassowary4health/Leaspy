import numpy as np
import torch

## TODO : Have a Abtract Attribute class
class Attributes:
    # TODO : Checker dès le début s'il est possible de conserver les attributes avec une méthode de gradient
    def __init__(self, dimension, source_dimension):
        # TODO : Supprimer dimension et source_dimension qui peuvent être déduits des values
        self.dimension = dimension
        self.source_dimension = source_dimension
        self.g = None  # g = exp(realizations['g']) tel que p0 = 1 / (1+exp(g))
        self.deltas = None  # deltas = [0, delta_2_realization, ..., delta_n_realization]
        self.orthonormal_basis = None
        self.mixing_matrix = None  # Matrix A tq w_i = A * s_i


        ## TODO : Add some individual attributes -> Optimization on the w_i = A * s_i

    def update(self, names_of_changed_values, values):
        """
        :param changed_parameters: list of variables that have been changed
        :param values: dictionary of {name: [value]}
        :return: None
        """
        self._check_names(names_of_changed_values)
        flag = self._flag_update(names_of_changed_values)
        if flag == 0:
            self._compute_g_and_deltas(values)
        elif flag == 1:
            self._compute_g(values)
        elif flag == 2:
            self._compute_deltas(values)
        elif flag == 3:
            self._compute_orthonormal_basis(values)
        elif flag == 4:
            self._compute_mixing_matrix(values)

    def _check_names(self, names_of_changed_values):
        for name in names_of_changed_values:
            if name not in ['all', 'g', 'deltas', 'betas', 'mean_tau', 'mean_xi']:
                raise ValueError("The name {} is not in the attributes that are used to be updated".format(name))

    def _flag_update(self, names_of_changed_values):
        if 'all' in names_of_changed_values:
            return 0
        if all(x in names_of_changed_values for x in ['g', 'deltas']):
            return 0
        if 'g' in names_of_changed_values:
            return 1
        if 'deltas' in names_of_changed_values:
            return 2
        if any(x in names_of_changed_values for x in ['mean_tau', 'mean_xi']):
            return 3
        if 'betas' in names_of_changed_values:
            return 4
        else:
            return 5

    def _compute_g_and_deltas(self, values):
        self.g = np.exp(values['g'])
        self.deltas = np.insert(values['deltas'], 0, 0)
        self._compute_orthonormal_basis(values)

    def _compute_g(self, values):
        self.g = np.exp(values['g'])
        self._compute_orthonormal_basis(values)

    def _compute_deltas(self, values):
        self.deltas = np.insert(values['deltas'], 0, 0)
        self._compute_orthonormal_basis(values)

    def _compute_orthonormal_basis(self, values):
        # Compute s
        # TODO : CHECK, CHECK AND RECHECK
        g = self.g
        v0 = np.exp(values['mean_xi'])
        E = np.exp(-self.deltas)
        A_ = 1. + g * E
        B_ = 1. / g + 1
        s = v0 * A_ * A_ * B_ * B_ / E

        # Compute Q
        e1 = np.repeat(0, self.dimension)
        e1[0] = 1
        a = (s+np.sign(s[0])*np.linalg.norm(s)*e1).reshape(1, -1)
        q_matrix = np.identity(self.dimension)-2*a.T.dot(a)/(a.dot(a.T))
        self.orthonormal_basis = q_matrix[:, 1:]

        # Compute Mixing matrix
        self._compute_mixing_matrix(values)
        # TODO : Test that the columns of the matrix are orthogonal to v0

    @staticmethod
    def _mixing_matrix_utils(linear_combination_values, matrix):
        return np.dot(matrix, linear_combination_values)

    def _compute_mixing_matrix(self, values):
        betas = np.array(values['betas'])
        self.mixing_matrix = torch.Tensor(self._mixing_matrix_utils(betas, self.orthonormal_basis))



