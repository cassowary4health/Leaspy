import torch


def Matrix(X,meta_settings):
"""
X est la donnée des points de controles un tensor de la forme (nb_pts,nb_dim), kernelname le nom du noyau à utiliser

cette fonction renvoie la matrice K_X
"""
    kernelname=meta_settings["kernelname"]
    sigma=meta_settings["sigma"]
    nb_ind,nb_tp,nb_dim=X.shape
    n=nb_ind*nb_tp
    X1=X.reshape((n,nb_dim))
    
    if kernelname="RBF":#le calcul est fait sans approximations
        sigma=meta_settings["sigma"]
        Z=X1.unsqueeze(-2)
        L1=np.ones((n,1))

        PA1=np.kron(L1,Z)
        PA2=np.transpose(PA1,axes=(1,0,2))

        Y=torch.from_numpy(PA1-PA2)

        K_X=torch.exp(-torch.norm(Y,dim=2)**2/(2*sigma**2))
    else:
        raise ValueError("Le nom de noyau est mauvais ! ")

    return K_X


def TransformationB(W,Control,meta_settings):
    """
    Prend en entrée la matrice des poids, et les points de contrôles "Control" ainsi que meta_settings pour avoir des
    informations sur le noyau, W de forme (nb_controle,nb_features). On renvoie la fonction associée pour update B.
    """
    if meta_settings[kernelname]="RBF":
        sigma=meta_settings["sigma"]
        def function(x):

            KK=torch.exp(-torch.norm(Control-x,dim=1)**2/(2*sigma**2))
            Fin=torch.matmul(KK,W).numpy()
            return torch.from_numpy(np.sum(Fin,axis=1))
        return function
    else:
        raise ValueError("Le nom de noyau est mauvais ! ")
