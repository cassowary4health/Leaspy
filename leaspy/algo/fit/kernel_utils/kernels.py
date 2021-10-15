import torch

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