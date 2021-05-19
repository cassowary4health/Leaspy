import torch
import numpy as np
import cvxpy as cp
from tqdm import tqdm

def dist(data, centers):
    distance = np.sum((np.array(centers) - data[:, None, :])**2, axis = 2)
    return distance

def kmeans_plus_plus(X, k,select=10**(-5)):
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
    
    
    i=0
    # Loop and select the remaining points
    while i<k:
        
        distance = dist(X, np.array(centers))
        
        if i == 0:
            pdf = distance/np.sum(distance)
            indexcour=np.random.choice(range(X.shape[0]), replace = False, p = pdf.flatten())
            centroid_new = X[indexcour]
            index.append(indexcour)
            i=i+1
        else:
            # Calculate the distance of each point from its nearest centroid
            dist_min = np.min(distance, axis = 1)
            
            pdf = dist_min/np.sum(dist_min)
# Sample one point from the given distribution
            indexcour=np.random.choice(range(X.shape[0]), replace = False, p = pdf)
            centroid_new = X[indexcour]
            distance = dist(centroid_new.reshape((1,-1)), np.array(centers))
            dist_min = np.min(distance, axis = 1)
            if dist_min>select:
                index.append(indexcour)
                i=i+1
            

            

        centers.append(centroid_new.tolist())
        
    index.sort()

    return index

def grid_control(Y,sig):
    """
    Parameters
    ----------
        Y: torch.tensor (nb_visite,dim) 
        k: int (number of control points)
    Returns:
    ---------
        control points uniformly distributed on [0,1] distant from sig aroud the points in Y
    



    """
    dim=Y.shape[1]
    

    nk=int(1/sig)
    grid1=torch.linspace(-0.5*sig,(nk+0.5)*sig,nk+1)
    L=[grid1]*dim
    T=torch.meshgrid(L)
    shape=list(T[0].shape)
    shape.append(dim)
    Per=[]
    for j in range(len(shape)):
        Per.append(j)
    Per[-1],Per[0]=Per[0],Per[-1]
    Z=torch.zeros(shape)
    Per=tuple(Per)
    Z=Z.permute(Per)
    for j in range(dim):
        Z[j]=T[j]
    Z=Z.permute(Per)

    Z=Z.reshape((-1,dim))
    index=[]
    for j in range(len(Z)):
        A=Z[j]
        dist=torch.norm(Y-A,dim=1)
        m=dist.min().item()
        if m<sig:
            index.append(j)
    X_con=Z[index]
    
    return X_con



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

def filtre_nan_homogene(XT,Y,mask):
    """
    Parameters
    ----------
        XT: torch tensor (nb_patient,nb_visit_max,dim) (latent points) et 
        Y:  torch tensoir (nb_patient,nb_visit_max,dim) (tensors)
        mask: torch tensor (nb_patient,nb_visit_max,dim) filled with 0 when there is no visit and 1 otherwise

    Return:
    -------
        X,Y torch tensor (nb_visit,dim)
    

    """
    maskbool=mask.bool().logical_not()
   
    Y[maskbool]=float("Nan")
    #XT n'a jamais de Nan car modèle génératif
    Select=(Y==Y).all(axis=2)#fonctionne bien voir notebook test pour se convaincre
    
    return XT[Select],Y[Select]

def filtre_nan_inhomogene(XT,Y,mask):
    """
    Prend en entrée XT (nb_patient,nb_visit_max,dim) et retourne X sous la forme (nb_visit,dim)

    Si un vecteur contient un Nan dans ses coordonnées on fait ????

    à voir ensemble Sam + PE

    """
    maskbool=mask.bool().logical_not()
    
    Y[maskbool]=float("Nan")
    
    Select=(Y==Y).any(axis=2)#si il n'y a pas d'observations
    
    return XT[Select],Y[Select]

def sigvalue(Y):
    nb_visit = len(Y)
    PA1 = Y.unsqueeze(0).repeat(nb_visit, 1, 1).permute(1, 0, 2)
    PA2 = Y.unsqueeze(0).repeat(nb_visit, 1, 1)
    PA3 = PA1 - PA2
    N=torch.norm(PA3, dim=-1)
    return torch.median(N)

#f



#Lorsqu'on rajoute un kernel il faut venir aussi rajouter la contraintes qui lui est associé
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
    def sobolev_kernel(X_points, meta_settings, X_control = None):
        '''

        Parameters
        ----------
        X_points : torch vector (N, features_dimension) or (n_ind, n_visits, features_dimension)
        with N points used for which we want kernel estimation
        meta_settings : dictionary with all settings
        X_control : optional, torch vector with control points used for kernel estimation

        Returns
        The matrix K(X_points_i, X_control_j) with the kernel K being the Sobolev with sigma given in the meta settings
        and ord, the order of the sobolev space
        If X_control is None, compute The matrix K(X_points_i, X_points_j)
        -------

        '''
        sigma = meta_settings["sigma"]
        order=meta_settings["order"]
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

        K_value = 1/(1+torch.norm(PA3, dim=-1) ** 2 /sigma ** 2)**order

        return K_value

    @staticmethod
    def get_kernel(name, meta_settings):
        if name in ["RBF", "gaussian"]:
            return lambda x, X_control=None: KernelFactory.rbf_kernel(x, meta_settings, X_control=X_control)
        if name in ["sobolev"]:
            return lambda x, X_control=None: KernelFactory.sobolev_kernel(x, meta_settings, X_control=X_control)
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


def transformation_B_compose(X_control, W, meta_settings,oldB):
    '''

    Parameters
    ----------
    X_control : torch vector (N, features_dimension) with N points used for which we want kernel estimation
    W : torch vector (N, features_dimension) of the weights associated with control points for kernel estimation
    meta_settings : dictionary with all settings
    oldB: last transformation tensor -> tensor

    Returns
    -------
    The function B computing the geodesics, taking as arguments :
    X_points : torch vector (n_points, features_dimension) of points for which we want to compute B(X_points)
    and returns the application of B to X_points of size (n_points, features_dimension)
    composed with oldB
    '''
    def func(X_pts):
        X_points=oldB(X_pts)
        return (X_points + compute_kernel_matrix(X_points, meta_settings, X_control = X_control) @ W).float()

    return func

def solver(X, Y, K_mul,K_con, dim,meta_settings):
    """
    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        K: torch.tensor(nb_visit,nb_control) kernel matrix associated to the kernel
        indices: list of int, enables to select the control points
        dim: int, dimension of the model
        meta_settings: dict, containing information about the kernel
    Returns
    -----
        W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """
    if "solver_quad" in meta_settings:
        if meta_settings["solver_quad"]=="cvxpy":
            W=optim_solver1(X, Y, K_mul, K_con, dim, meta_settings)
            
        else:
            W=optim_solver3(X, Y, K_mul, K_con, dim, meta_settings)
    else:
        #W=optim_solver2(X, Y, K, indices, dim, meta_settings)
        W=optim_solver3(X, Y, K_mul, K_con, dim, meta_settings)
    return torch.from_numpy(W).to(torch.float32)



def kernelreg(meta_settings,dim):
    """
    Parameters
    -----
        meta_settings: Dict
        dim: int, dimension of the model
    Returns
    -----
        concon: Float, the constraint to respect associated to the kernel selected

    """
    if meta_settings["kernel_name"]in ["RBF", "gaussian"]:
        concon=meta_settings["sigma"]**2/dim
        
    elif meta_settings["kernel_name"]in ["sobolev"]:
        concon=meta_settings["sigma"]**2/(dim*meta_settings["order"])
    else:
        raise NotImplementedError("Your kernel {} is not available".format(name))
    return concon
    
def optim_solver1(X, Y, K, indices, dim, meta_settings):
    """
    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        K: torch.tensor(nb_visit,nb_control) kernel matrix associated to the kernel
        indices: list of int, enables to select the control points
        dim: int, dimension of the model
        meta_settings: dict, containing information about the kernel
    Returns
    -----
        W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """

    convalue= kernelreg(meta_settings,dim)
    W = cp.Variable((K.shape[1], dim))
    K_ = cp.Parameter((K.shape[1], K.shape[1]), PSD=True)
    K_.value = K[indices].detach().numpy()
    X_ = X.detach().numpy()
    Y_ = Y.detach().numpy()

    constraints = [cp.atoms.sum([cp.atoms.quad_form(W[:,k], K_) for k in range(dim)]) <= convalue]
    prob = cp.Problem(cp.Minimize(cp.atoms.norm(Y - (X + K.detach().numpy()@W),"fro")), constraints)
    
    prob.solve()
    
    return W.value

import time
import scipy.linalg as LA
def optim_solver3(X, Y, K_mul,K_con, dim, meta_settings):
    """
    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        indices: list of int, enables to select the control points
        dim: int, dimension of the model
        meta_settings: dict, containing information about the kernel
    Returns
    -----
         W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """
    t1=time.clock()
    KCC1 = K_con.detach().numpy()
    KG1=K_mul.detach().numpy()#bon même avec nan
    nb_con=KCC1.shape[0]
    dim=Y.shape[1]
    
    
    X_ = X.detach().numpy()
    Y_ = Y.detach().numpy()
    nb_patient=Y_.shape[0]
    #X_=X_.transpose().reshape((-1,))#On a d'abord toutes les coordoonées de dim1 puis 2 etc..
    #Y_=Y_.transpose().reshape((-1,))
    
    L_const=[]
    ind=[]
    KKK=[]
    for j in range(dim):
        
        for i in range(nb_patient):
            if Y_[i,j]==Y_[i,j]:
                L_const.append(Y_[i,j]-X_[i,j])
                ind.append((i,j))
                ZK=[0]*dim*nb_con
                ZK[j*nb_con:(j+1)*nb_con]=list(KG1[i])
                KKK.append(ZK)

          
    Const=np.array(L_const)
    KG=np.array(KKK)
    print("KG")
    print(KG.shape)
    print("Const")
    print(Const.shape)

    #Si KG matrice rectangulaire par bloc ok, Const ok
    DD=KG.transpose()@Const
    #KG doit être de taille (n_visit,control)


    Kred=KG.transpose()@KG
    lambd=1
    lambdmin=1
    
    KCC=np.kron(np.eye(dim,dtype=int),KCC1)
    Mat=lambd*KCC+Kred
    
    w,V=LA.eigh(Mat)
    delta = np.abs(Mat - (V * w).dot(V.T))
    print("erreur projection")
    print(LA.norm(delta, ord=2))

    W=np.linalg.solve(Mat,DD)

    contrainte=kernelreg(meta_settings,dim)-2*10**(-3)

    
    g=lambda w: w.transpose()@KCC@w

    while g(W)>contrainte:
        print(lambd)
        lambd,lambdmin=lambd*2,lambd
        Mat=lambd*KCC+Kred
        W=np.linalg.solve(Mat,DD)
    #rajouter ensuite une recherche dicotomique du lambda optimal
    f=lambda l: np.linalg.solve(l*KCC+Kred,DD).transpose()@KCC@np.linalg.solve(l*KCC+Kred,DD)
    lopt=dicho(lambdmin,lambd,contrainte,f)
    Mat=lopt*KCC+Kred
    W=np.linalg.solve(Mat,DD)
    t2=time.clock()
    print("temps opti quadra")
    print(t2-t1)
    print("error least square")
    print(np.linalg.norm(Const-KG@W,2)/len(Const))
    W=W.reshape((dim,nb_con)).transpose()
    
    return W

def optim_solver1(X, Y, K,K_con, dim, meta_settings):
    """
    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        K: torch.tensor(nb_visit,nb_control) kernel matrix associated to the kernel
        indices: list of int, enables to select the control points
        dim: int, dimension of the model
        meta_settings: dict, containing information about the kernel
    Returns
    -----
        W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """

    convalue= kernelreg(meta_settings,dim)
    W = cp.Variable((K.shape[1], dim))
    K_ = cp.Parameter((K.shape[1], K.shape[1]), PSD=True)
    K_.value = K[indices].detach().numpy()
    X_ = X.detach().numpy()
    Y_ = Y.detach().numpy()

    constraints = [cp.atoms.sum([cp.atoms.quad_form(W[:,k], K_) for k in range(dim)]) <= convalue]
    prob = cp.Problem(cp.Minimize(cp.atoms.norm(Y - (X + K.detach().numpy()@W),"fro")), constraints)
    
    prob.solve()
    
    return W.value

import time
import scipy.linalg as LA
def optim_solver2(X, Y, K_mul, K_con, dim, meta_settings):
    """
    Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        indices: list of int, enables to select the control points
        dim: int, dimension of the model
        meta_settings: dict, containing information about the kernel
    Returns
    -----
         W: numpy.array (nb_control,dim), the solution of the constrained quadratic problem

    """
    t1=time.clock()
    KCC = K_con.detach().numpy()
    KG=K_mul.detach().numpy()#bon même avec nan

    Kred=KG.transpose()@KG
    
    X_ = X.detach().numpy()
    Y_ = Y.detach().numpy()

    Const=Y_-X_
    DD=KG.transpose()@Const
    lambd=1
    lambdmin=1
    Mat=lambd*KCC+Kred
    
    w,V=LA.eigh(Mat)
    delta = np.abs(Mat - (V * w).dot(V.T))
    print("erreur projection")
    print(LA.norm(delta, ord=2))

    W=np.linalg.solve(Mat,DD)

    contrainte=kernelreg(meta_settings,dim)-2*10**(-3)


    g=lambda w: np.trace(w.transpose()@KCC@w)

    while g(W)>contrainte:
        print(lambd)
        lambd,lambdmin=lambd*2,lambd
        Mat=lambd*KCC+Kred
        W=np.linalg.solve(Mat,DD)
    #rajouter ensuite une recherche dicotomique du lambda optimal
    f=lambda l: np.trace(np.linalg.solve(l*KCC+Kred,DD).transpose()@KCC@np.linalg.solve(l*KCC+Kred,DD))
    lopt=dicho(lambdmin,lambd,contrainte,f)
    Mat=lopt*KCC+Kred
    W=np.linalg.solve(Mat,DD)
    t2=time.clock()
    print("temps opti quadra")
    print(t2-t1)
    return W


def dicho(a,b,c,f,err=10**(-3)):
    """
     Parameters
    -----
        X: torch.tensor (nb_visit,dim) The points of the lattent space
        Y: torch.tensor (nb_visit,dim)  to match with the kernel
        indices: list of int, enables to select the control points
        dim: int, dimension of the model
        meta_settings: dict, containing information about the kernel
    Returns
    -----
        concon: Float, the constraint to respect associated to the kernel selected

    """
    mi=a
    ma=b
    pivot=(mi+ma)/2
    dec=abs(f(pivot)-c)
    k=0
    while dec>err and k<20:
        k=k+1
        comp=f(pivot)
        if comp>c:
            mi=pivot
        else:
            ma=pivot
        pivot=(mi+ma)/2
        dec=abs(f(pivot)-c)
    return pivot
        
