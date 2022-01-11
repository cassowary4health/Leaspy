import torch
import numpy as np
import scipy.linalg as LA

def compute_kernel_matrix(X_points, meta_settings, X_control=None):
    '''

    Parameters
    ----------
    X_points : torch vector (N, features_dimension) or (n_ind, n_visits, features_dimension)
    with N points used for which we want kernel estimation
    meta_settings : dictionary with all settings
    X_control : optional, torch vector with control points used for kernel estimation

    Returns
    The matrix K(X_points_i, X_control_j) with the kernel K being the RBF with sigma given in the meta settings
    If X_control is None, compute The matrix K(X_points_i, X_points_j)
    -------

    '''

    sigma = meta_settings["sigma"]
    if X_control is None:
        X_control = X_points
    k = len(X_control)

    if len(X_points.shape) == 1:
        X_points = X_points.unsqueeze(0)

    if len(X_points.shape) == 2:
        nb_visit = len(X_points)

        lines = X_points.unsqueeze(0).repeat(k, 1, 1).permute(1, 0, 2)
        columns = X_control.unsqueeze(0).repeat(nb_visit, 1, 1)

    else:
        n_ind = X_points.shape[0]
        n_visits = X_points.shape[1]

        lines = X_points.unsqueeze(0).repeat(k, 1, 1, 1).permute(1, 2, 0, 3)
        columns = X_control.unsqueeze(0).unsqueeze(0).repeat(n_ind, n_visits, 1, 1)

    # Basis for any kernel of the form k(x, y) = g(x -y)

    differences = lines - columns

    # Now comes the actual kernel :
    name = meta_settings["kernel_name"]

    if name.lower() in ["rbf", "gaussian"]:
        kernel_value = torch.exp(-torch.norm(differences, dim=-1) ** 2 / (2 * sigma ** 2))

    elif name.lower() in ["sobolev", "t-student"]:
        order = meta_settings["order"]
        kernel_value = 1 / (1 + torch.norm(differences, dim=-1) ** 2 / sigma ** 2) ** order
    else:
        raise NotImplementedError("Your kernel {} is not available, currently only {} are available".format(name, ["gaussian", "sobolev"]))

    return kernel_value

def apply_diffeomorphism(X_control, weights, meta_settings):
    '''

    Parameters
    ----------
    X_control : torch vector (N, features_dimension) with N control points used for kernel estimation
    weights : torch vector (N, features_dimension) of the weights associated with control points for kernel estimation
    meta_settings : dictionary with all settings

    Returns
    -------
    The diffeomorphism phi computing the geodesics, taking as arguments :
    X_points : torch vector (n_points, features_dimension) of points for which we want to compute phi(X_points)
    and returns the application of phi to X_points of size (n_points, features_dimension)
    '''

    return lambda X_points : (X_points + compute_kernel_matrix(X_points, meta_settings, X_control = X_control) @ weights).float()


def kernel_constraint(meta_settings, dim):
    """
    Parameters
    -----
        meta_settings: Dict
        dim: int, dimension of the model
    Returns
    -----
        constraint: Float, the constraint to respect associated to the kernel selected

    """
    if meta_settings["kernel_name"].lower() in ["rbf", "gaussian"]:
        constraint = meta_settings["sigma"] ** 2 / dim

    elif meta_settings["kernel_name"].lower() in ["sobolev", "t-student"]:
        constraint = meta_settings["sigma"] ** 2 / (dim * meta_settings["order"])
    else:
        raise NotImplementedError("Your kernel {} is not available".format(meta_settings["kernel_name"]))
    return constraint

def kernel_optimal_window(dataset):
    """
    Computes the optimal empirical sigma for the gaussian and sobolev kernel according to data
    Parameters
    ----------
    dataset: class:`.Dataset`
            Contains the subjects' observations in torch format to speed up computation.

    Returns
    -------
    sigma: class:`torch.Tensor`
        The value computed
    """
    nb_visit = dataset.n_visits
    rows = dataset.values.unsqueeze(0).repeat(nb_visit, 1, 1).permute(1, 0, 2)
    columns = dataset.values.unsqueeze(0).repeat(nb_visit, 1, 1)
    distances = rows - columns
    norms = torch.norm(distances, dim=-1)
    return torch.median(norms)


def control_points_grid(dataset, sigma):
    """
    Parameters
    ----------
        dataset: class:`.Dataset`
            Contains the subjects' observations in torch format to speed up computation.
        sigma: int (number of control points)
    Returns:
    ---------
        control points uniformly distributed on [0,1] distant from sig aroud the points in Y




    """
    dim = Y.shape[1]

    nk = int(1 / sig)
    grid1 = torch.linspace(-0.5 * sig, (nk + 0.5) * sig, nk + 1)
    L = [grid1] * dim
    T = torch.meshgrid(L)
    shape = list(T[0].shape)
    shape.append(dim)
    Per = []
    for j in range(len(shape)):
        Per.append(j)
    Per[-1], Per[0] = Per[0], Per[-1]
    Z = torch.zeros(shape)
    Per = tuple(Per)
    Z = Z.permute(Per)
    for j in range(dim):
        Z[j] = T[j]
    Z = Z.permute(Per)

    Z = Z.reshape((-1, dim))
    index = []
    for j in range(len(Z)):
        A = Z[j]
        dist = torch.norm(Y - A, dim=1)
        m = dist.min().item()
        if m < sig:
            index.append(j)
    X_con = Z[index]

    return X_con


def dist(data, centers):
    distance = np.sum((np.array(centers) - data[:, None, :]) ** 2, axis=2)
    return distance


def kmeans_plus_plus(X, k, select=10 ** (-5)):
    '''Initialize one point at random.
    loop for k - 1 iterations:
        Next, calculate for each point the distance of the point from its nearest center. Sample a point with a
        probability proportional to the square of the distance of the point from its nearest center.'''
    centers = []
    index = []

    # Sample the first point
    initial_index = np.random.choice(range(X.shape[0]), )
    index.append(initial_index)
    centers.append(X[initial_index, :].tolist())

    i = 0
    # Loop and select the remaining points
    while i < k:

        distance = dist(X, np.array(centers))

        if i == 0:
            pdf = distance / np.sum(distance)
            indexcour = np.random.choice(range(X.shape[0]), replace=False, p=pdf.flatten())
            centroid_new = X[indexcour]
            index.append(indexcour)
            i = i + 1
        else:
            # Calculate the distance of each point from its nearest centroid
            dist_min = np.min(distance, axis=1)

            pdf = dist_min / np.sum(dist_min)
            # Sample one point from the given distribution
            indexcour = np.random.choice(range(X.shape[0]), replace=False, p=pdf)
            centroid_new = X[indexcour]
            distance = dist(centroid_new.reshape((1, -1)), np.array(centers))
            dist_min = np.min(distance, axis=1)
            if dist_min > select:
                index.append(indexcour)
                i = i + 1

        centers.append(centroid_new.tolist())

    index.sort()

    return index

def sub_sampling(X,k):
    """
    Parameters
    ----------
        X: torch.tensor (nb_visite,dim)
        k: int (number of control points)
    Returns:
    ---------
        index list of int, the indices of control points selected in X
    We use kmeans_plus_plus to take points not so close in order to have a kernel matrix well specified
    """
    index = kmeans_plus_plus(X.numpy(), k)
    return index


def solver(X, Y, K, K_control, meta_settings):
    """
    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        K : torch.tensor(nb_visit,nb_control) kernel matrix associated to the kernel
        K_control : torch.tensor(nb_control, nb_control) kernel matrix for control points only
        indices: list of int, enables to select the control points
        dim: int, dimension of the model
        meta_settings: dict, containing information about the kernel
    Returns
    -----
        W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """
    if "solver" in meta_settings:
        if meta_settings["solver"] == "cvxpy":
            W = cvx_solver(X, Y, K, K_control, meta_settings)
            return torch.from_numpy(W).to(torch.float32)
    W = optim_solver(X, Y, K, K_control, meta_settings)
    return torch.from_numpy(W).to(torch.float32)


def cvx_solver(X, Y, K, K_control, meta_settings):
    """
    Not robust to missing values

    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        K: torch.tensor(nb_visit,nb_control) kernel matrix associated to the kernel
        K_control : torch.tensor(nb_control, nb_control) kernel matrix for control points only
        meta_settings: dict, containing information about the kernel
    Returns
    -----
        W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """
    import cvxpy as cp
    dim = X.shape[1]
    constraint_value = kernel_constraint(meta_settings, dim)
    W = cp.Variable((K.shape[1], dim))
    K_ = cp.Parameter((K.shape[1], K.shape[1]), PSD=True)
    K_.value = K_control.detach().numpy()
    X_ = X.detach().numpy()
    Y_ = Y.detach().numpy()

    constraints = [cp.atoms.sum([cp.atoms.quad_form(W[:, k], K_) for k in range(dim)]) <= constraint_value]
    prob = cp.Problem(cp.Minimize(cp.atoms.norm(Y - (X + K.detach().numpy() @ W), "fro")), constraints)

    prob.solve()

    return W.value

def optim_solver(X, Y, K, K_control, meta_settings):
    """
    robust to missing values

    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        K: torch.tensor(nb_visit,nb_control,dim) kernel matrix associated to the kernel
        K_control : torch.tensor(nb_control, nb_control,dim) kernel matrix for control points only
        meta_settings: dict, containing information about the kernel
    Returns
    -----
         W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """

    K_c = K_control.detach().numpy()
    K_X = K.detach().numpy()
    nb_control_points = K_c.shape[0]

    X_ = X.detach().numpy()
    Y_ = Y.detach().numpy()
    dim = Y_.shape[1]
    nb_patient = Y_.shape[0]

    L_const = []
    ind = []
    KKK = []

    # Flattening while removing Nan
    for j in range(dim):
        for i in range(nb_patient):
            if Y_[i, j] == Y_[i, j]:
                L_const.append(Y_[i, j] - X_[i, j])
                ind.append((i, j))
                ZK = [0] * dim * nb_control_points
                ZK[j * nb_control_points:(j + 1) * nb_control_points] = list(K_X[i])
                KKK.append(ZK)

    Const = np.array(L_const)
    KG = np.array(KKK)

    # Si KG matrice rectangulaire par bloc ok, Const ok
    DD = KG.transpose() @ Const
    # KG doit être de taille (n_visit,control)

    Kred = KG.transpose() @ KG
    lambd = 1
    lambdmin = 1

    KCC = np.kron(np.eye(dim, dtype=int), K_c)
    Mat = lambd * KCC + Kred

    w, V = LA.eigh(Mat)
    delta = np.abs(Mat - (V * w).dot(V.T))

    W = np.linalg.solve(Mat, DD)

    constraint = kernel_constraint(meta_settings, dim) - 2 * 10 ** (-3)

    g = lambda w: w.transpose() @ KCC @ w

    while g(W) > constraint and lambd < 10 ** (6):
        lambd, lambdmin = lambd * 2, lambd
        Mat = lambd * KCC + Kred
        W = np.linalg.solve(Mat, DD)
    # rajouter ensuite une recherche dicotomique du lambda optimal
    f = lambda l: np.linalg.solve(l * KCC + Kred, DD).transpose() @ KCC @ np.linalg.solve(l * KCC + Kred, DD)
    lopt = dicho(lambdmin, lambd, contrainte, f)
    Mat = lopt * KCC + Kred
    W = np.linalg.solve(Mat, DD)
    W = W.reshape((dim, nb_control_points)).transpose()

    return W

def weights_optimization(X, Y, parameters, control_points):
    """

    Parameters
    ----------
    X : model estimation for observations
    Y : observations
    mask : mask for NaN
    parameters : dictionnary with required parameters about kernel estimation
    control_points : selected control points for kernel estimation

    Returns
    -------
    weights : the weights associated to the control points
    """



    # We take the matric associated to control points to perform the kernel estimation
    objective = compute_kernel_matrix(X, parameters, control_points)
    # (nb_visit,N_c(sig))
    # We take the matrix used to compute the constraints
    constraint = compute_kernel_matrix(control_points, parameters, control_points)
    # (N_c(sig),N_c(sig))

    # We compute the kernel weights associated to the new diffeomorphism
    weights = solver(X, Y, objective, constraint, parameters)

    return weights
