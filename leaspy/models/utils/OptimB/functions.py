import torch


def dist(data, centers):
    distance = np.sum((np.array(centers) - data[:, None, :])**2, axis = 2)
    return distance
def kmeans_plus_plus(X, k, pdf_method = True):
    '''Initialize one point at random.
    loop for k - 1 iterations:
        Next, calculate for each point the distance of the point from its nearest center. Sample a point with a 
        probability proportional to the square of the distance of the point from its nearest center.'''
    centers = []
    
    
    # Sample the first point
    initial_index = np.random.choice(range(X.shape[0]), )
    centers.append(X[initial_index, :].tolist())
    
    print('max: ', np.max(np.sum((X - np.array(centers))**2)))
    
    # Loop and select the remaining points
    for i in range(k - 1):
        print(i)
        distance = dist(X, np.array(centers))
        
        if i == 0:
            pdf = distance/np.sum(distance)
            centroid_new = X[np.random.choice(range(X.shape[0]), replace = False, p = pdf.flatten())]
        else:
            # Calculate the distance of each point from its nearest centroid
            dist_min = np.min(distance, axis = 1)
            if pdf_method == True:
                pdf = dist_min/np.sum(dist_min)
# Sample one point from the given distribution
                centroid_new = X[np.random.choice(range(X.shape[0]), replace = False, p = pdf)]
            else:
                index_max = np.argmax(dist_min, axis = 0)
                centroid_new = X[index_max, :]

        centers.append(centroid_new.tolist())
        
    return np.array(centers)

def Sub_sampling(X,k):
    """
    Prend X le tensor (nb_visite,dim) et sélectionne k points bien espacé renvoyé dans un tensor (k,dim)


    """
    Center=kmeans_plus_plus(X.numpy(), k)
    return torch.from_numpy(Center)

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

def FiltreNanHomogène(XT):
    """
    Prend en entrée XT (nb_patient,nb_visit_max,dim) et retourne X sous la forme (nb_visit,dim)

    Si un vecteur contient un Nan dans ses coordonnées on le retire

    """


    Select=(XT==XT).all(axis=2)#fonctionne bien voir notebook test pour se convaincre
    
    return XT[Select]

def FiltreNanInHomogène(XT):
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











def Matrix(X,meta_settings):
"""
X est la donnée des points de controles un tensor de la forme (nb_pts,nb_dim), kernelname le nom du noyau à utiliser

cette fonction renvoie la matrice K_X
"""
    kernelname=meta_settings["kernelname"]
    sigma=meta_settings["sigma"]
    
    
    if kernelname=="RBF":#le calcul est fait sans approximations
        sigma=meta_settings["sigma"]
        Z=X.unsqueeze(-2).numpy()
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
    if meta_settings[kernelname]=="RBF":
        sigma=meta_settings["sigma"]
        def function(x):

            KK=torch.exp(-torch.norm(Control-x,dim=1)**2/(2*sigma**2))
            Fin=torch.matmul(KK,W).numpy()
            return torch.from_numpy(np.sum(Fin,axis=1))
        return function
    else:
        raise ValueError("Le nom de noyau est mauvais ! ")
