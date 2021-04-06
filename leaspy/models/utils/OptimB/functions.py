import torch
import numpy as np

def dist(data, centers):
    distance = np.sum((np.array(centers) - data[:, None, :])**2, axis = 2)
    return distance

def kmeans_plus_plus(X, k):
    '''Initialize one point at random.
    loop for k - 1 iterations:
        Next, calculate for each point the distance of the point from its nearest center. Sample a point with a 
        probability proportional to the square of the distance of the point from its nearest center.'''
    centers = []
    index=[]
    
    # Sample the first point
    initial_index = np.random.choice(range(X.shape[0]), )
    index.append(initial_index)
    centers.append(X[initial_index, :].tolist())
    
    
    
    # Loop and select the remaining points
    for i in range(k - 1):
        
        distance = dist(X, np.array(centers))
        
        if i == 0:
            pdf = distance/np.sum(distance)
            indexcour=np.random.choice(range(X.shape[0]), replace = False, p = pdf.flatten())
            centroid_new = X[indexcour]
            index.append(indexcour)
        else:
            # Calculate the distance of each point from its nearest centroid
            dist_min = np.min(distance, axis = 1)
            
            pdf = dist_min/np.sum(dist_min)
# Sample one point from the given distribution
            indexcour=np.random.choice(range(X.shape[0]), replace = False, p = pdf)
            centroid_new = X[indexcour]
            index.append(indexcour)
            

        centers.append(centroid_new.tolist())
        
    index.sort()

    return index

def sub_sampling(X,k):
    """
    Prend X le tensor (nb_visite,dim) et sélectionne k points bien espacé renvoyé dans un tensor (k,dim)

    testé approuvé


    """
    index = kmeans_plus_plus(X.numpy(), k)
    return index

"""
Pour filtrer les Nan, il faut réfléchir. Si on a des Nan réparti de manière inhomogène par exemple : [[Nan,Nan],[1,2]]
on peut se permettre de supprimer le vecteur avec que des Nan, et supposer qu'on a plus de Nan ailleurs, on supprime aussi
le label Y_i,j associé pour déterminer les poids.
En revanche, si, comme dans la plupart des cas, on a des Nan répartis de manière inhomogène[[1,Nan],[NaN,2]], on se retrouve
pas bien. En effet il ne suffit pas s'intéresser à chacune des composantes pour apprendre le noyau car si on veut calculer
un noyau on est amener à calculer |x_i-x_j|, dans cette expression on se retrouve possiblement à avoir x_i=(Nan,1) et x_j=(1,2)
dans ce cas, on peut supprimer les calculs avec Nan de telsorte que |x_i-x_j|=|1-2+ 0|. Cependant dans le cas x_i=[1,Nan] et
x_j=[NaN,2], c'est inexploitable, on se retrouverai avec |x_i-x_j|=0


"""

def filtre_nan_homogene(XT,Y):
    """
    Prend en entrée XT (nb_patient,nb_visit_max,dim) et retourne X sous la forme (nb_visit,dim)

    Si un vecteur contient un Nan dans ses coordonnées on le retire

    testé approuvé

    """


    Select=((XT==XT).all(axis=2))*(Y==Y).all(axis=2)#fonctionne bien voir notebook test pour se convaincre
    
    return XT[Select],Y[Select]

def filtre_nan_inhomogene(XT):
    """
    Prend en entrée XT (nb_patient,nb_visit_max,dim) et retourne X sous la forme (nb_visit,dim)

    Si un vecteur contient un Nan dans ses coordonnées on fait ????

    à voir ensemble Sam + PE

    """
    Select1=(XT==XT).all(axis=2)

    PartHomo=XT[Select1]
    
    Select2=(XT!=XT).any(axis=2)*(XT==XT).any(axis=2)#voir test pour se convaincre

    PartInHomo=XT[Select2]
    
    #Pour l'instant on ne fait rien de la partie inhomogène
    return PartHomo







class KernelFactory:

    @staticmethod
    def rbf_kernel(X_points, meta_settings, X_control = None):
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

        ndim = len(X_points.shape)

        if ndim == 2:
            nb_visit = len(X_points)

            PA1 = X_points.unsqueeze(0).repeat(k, 1, 1).permute(1, 0, 2)
            PA2 = X_control.unsqueeze(0).repeat(nb_visit, 1, 1)

        elif ndim == 3:
            n_ind = X_points.shape[0]
            n_visits = X_points.shape[1]

            PA1 = X_points.unsqueeze(0).repeat(k, 1, 1, 1).permute(1, 2, 0, 3)
            PA2 = X_control.unsqueeze(0).unsqueeze(0).repeat(n_ind, n_visits, 1, 1)

        PA3 = PA1 - PA2

        K_value = torch.exp(-torch.norm(PA3, dim=-1) ** 2 / (2 * sigma ** 2))

        return K_value

    @staticmethod
    def get_kernel(name, meta_settings):
        if name in ["RBF", "gaussian"]:
            return lambda x, X_control=None: KernelFactory.rbf_kernel(x, meta_settings, X_control=X_control)
        else:
            raise NotImplementedError("Your kernel {} is not available".format(name))




def compute_kernel_matrix(X_points, meta_settings, X_control = None):
    '''

        Parameters
        ----------
        X_points : torch vector (N, features_dimension) or (n_ind, n_visits, features_dimension)
        with N points used for which we want kernel estimation
        meta_settings : dictionary with all settings
        X_control : optional, torch vector with control points used for kernel estimation

        Returns
        The matrix K(x_i, x_j) with the kernel K specified by meta_settings["kernelname"]
        -------

        '''
    kernel_name = meta_settings["kernel_name"]
    kernel = KernelFactory.get_kernel(kernel_name, meta_settings)

    return kernel(X_points, X_control = X_control)


def transformation_B(X_control, W, meta_settings):
    '''

    Parameters
    ----------
    X_control : torch vector (N, features_dimension) with N points used for which we want kernel estimation
    W : torch vector (N, features_dimension) of the weights associated with control points for kernel estimation
    meta_settings : dictionary with all settings

    Returns
    -------
    The function B computing the geodesics, taking as arguments :
    X_points : torch vector (n_points, features_dimension) of points for which we want to compute B(X_points)
    and returns the application of B to X_points of size (n_points, features_dimension)
    '''

    return lambda X_points : (X_points + compute_kernel_matrix(X_points, meta_settings, X_control = X_control) @ W).float()

def solver(MatValue,MatContrainte,Constante,meta_settings):
    """
    Prend en entrée les paramètres du problème permettant de reconstruire le problème quadratique 

    """
    Hess=torch.matmul(MatValue.transpose(0,1),MatValue)/2

    LinConstnate=torch.matmul(MatValue.transpose(0,1),Constante)






    raise ValueError(("not implemented"))
