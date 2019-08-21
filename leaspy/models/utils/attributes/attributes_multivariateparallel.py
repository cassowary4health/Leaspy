import torch


## TODO 1 : Have a Abtract Attribute class
## TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class Attributes_MultivariateParallel:

    def __init__(self, dimension, source_dimension):
        self.dimension = dimension
        self.source_dimension = source_dimension
        self.g = None  # g = exp(realizations['g']) tel que p0 = 1 / (1+exp(g))
        self.deltas = None  # deltas = [0, delta_2_realization, ..., delta_n_realization]
        self.v0 = None  # v0 is a scalar value, which corresponds to the the first dimension of the velocity vector
        self.betas = None
        self.orthonormal_basis = None
        self.mixing_matrix = None  # Matrix A tq w_i = A * s_i

    def update(self, names_of_changed_values, values):
        self._check_names(names_of_changed_values)

        compute_g = False
        compute_v0 = False
        compute_deltas = False
        compute_betas = False

        for name in names_of_changed_values:
            if name == 'g':
                compute_g = True
            elif name == 'deltas':
                compute_deltas = True
            elif name == 'betas':
                compute_betas = True
            elif name == 'xi_mean':
                compute_v0 = True
            elif name == 'all':
                compute_g = True
                compute_deltas = True
                compute_v0 = True
                compute_betas = True

        if compute_g: self._compute_g(values)
        if compute_deltas: self._compute_deltas(values)
        if compute_v0: self._compute_v0(values)
        if compute_betas: self._compute_betas(values)

        if compute_g or compute_deltas or compute_v0:
            self._compute_orthonormal_basis()
        if compute_g or compute_deltas or compute_v0 or compute_betas:
            self._compute_mixing_matrix()

    def _check_names(self, names_of_changed_values):
        for name in names_of_changed_values:
            if name not in ['g', 'deltas', 'betas', 'xi_mean', 'all']:
                raise ValueError("The name {} is not in the attributes that are used to be updated".format(name))

    def _compute_v0(self, values):
        self.v0 = torch.exp(torch.Tensor([values['xi_mean']]))

    def _compute_g(self, values):
        self.g = torch.exp(values['g'])

    def _compute_deltas(self, values):
        self.deltas = torch.cat((torch.Tensor([0]), values['deltas']))

    def _compute_betas(self, values):
        if self.source_dimension == 0:
            return
        self.betas = torch.Tensor(values['betas'])

    def _compute_orthonormal_basis(self):
        if self.source_dimension == 0:
            return

        # Compute s
        # TODO : CHECK, CHECK AND RECHECK
        # TODO : Test that the columns of the matrix are orthogonal to v0
        g = self.g
        v0 = self.v0
        E = torch.exp(-self.deltas)
        A_ = 1. + g * E
        B_ = 1. / g + 1
        s = v0 * A_ * A_ * B_ * B_ / E

        # Compute Q
        e1 = torch.zeros((self.dimension))
        e1[0] = 1
        a = torch.sign(s[0]) * torch.norm(s)
        a = a * e1
        a = s + a
        a = a.reshape(1, -1)
        #a = (s+np.sign(s[0])*torch.norm(s)*e1).reshape(1, -1)

        q_matrix = -2 * a.transpose(0, 1) * a
        q_matrix = q_matrix / torch.mm(a, a.transpose(0, 1))
        q_matrix = q_matrix + torch.eye(self.dimension)
        #q_matrix = np.identity(self.dimension)-2*a.transpose(0, 1).dot(a)/(a.dot(a.transpose(0, 1)))
        self.orthonormal_basis = q_matrix[:, 1:]

    @staticmethod
    def _mixing_matrix_utils(linear_combination_values, matrix):
        return torch.mm(matrix, linear_combination_values)

    def _compute_mixing_matrix(self):
        if self.source_dimension == 0:
            return

        self.mixing_matrix = torch.Tensor(self._mixing_matrix_utils(self.betas, self.orthonormal_basis))



