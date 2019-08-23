import torch


## TODO 1 : Have a Abtract Attribute class
## TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class Attributes_Logistic:

    def __init__(self, dimension, source_dimension):

        self.dimension = dimension
        self.source_dimension = source_dimension
        self.g = None  # g is a vector such that p0 = 1 / (1+exp(g)) where p0 is the position vector
        self.v0 = None  # v0 is the vector of velocities
        self.orthonormal_basis = None
        self.mixing_matrix = None  # Matrix A tq w_i = A * s_i

    def update(self, names_of_changed_values, values):
        self._check_names(names_of_changed_values)

        compute_g = False
        compute_v0 = False
        compute_betas = False

        for name in names_of_changed_values:
            if name == 'g':
                compute_g = True
            elif name == 'v0':
                compute_v0 = True
            elif name == 'betas':
                compute_betas = True
            elif name == 'all':
                compute_g = True
                compute_v0 = True
                compute_betas = True

        if compute_g: self._compute_g(values)
        if compute_v0: self._compute_v0(values)
        if compute_betas: self._compute_betas(values)

        if compute_g or compute_v0:
            self._compute_orthonormal_basis()
        if compute_g or compute_v0 or compute_betas:
            self._compute_mixing_matrix(values)

    def _check_names(self, names_of_changed_values):
        for name in names_of_changed_values:
            if name not in ['all', 'g', 'v0', 'betas']:
                raise ValueError("The name {} is not in the attributes that are used to be updated".format(name))

    def _compute_g(self, values):
        self.g = torch.exp(values['g'])

    def _compute_v0(self, values):
        self.v0 = torch.exp(values['v0'])

    def _compute_betas(self, values):
        if self.source_dimension == 0:
            return
        self.betas = torch.Tensor(values['betas'])

    def _compute_orthonormal_basis(self):
        if self.source_dimension == 0:
            return
        p = 1/(1+self.g)
        G_p = 1/(p*(1-p))**2
        s = G_p*self.v0

        # Compute Q
        e1 = torch.zeros(self.dimension)
        e1[0] = 1
        a = (s+torch.sign(s[0])*torch.norm(s)*e1).reshape(1, -1)

        q_matrix = -2 * a.transpose(0, 1) * a
        q_matrix = q_matrix / torch.mm(a, a.transpose(0, 1))
        q_matrix = q_matrix + torch.eye(self.dimension)

        self.orthonormal_basis = q_matrix[:, 1:]

    @staticmethod
    def _mixing_matrix_utils(linear_combination_values, matrix):
        return torch.mm(matrix, linear_combination_values)

    def _compute_mixing_matrix(self, values):
        if self.source_dimension == 0:
            return
        self.mixing_matrix = torch.Tensor(self._mixing_matrix_utils(self.betas, self.orthonormal_basis))

        #if (torch.mm(torch.Tensor(values['v0']), self.mixing_matrix).sum()>1e-5):
        #    print("coucou, bug ??? {}".format(torch.mm(torch.Tensor(values['v0']), self.mixing_matrix).sum()))


