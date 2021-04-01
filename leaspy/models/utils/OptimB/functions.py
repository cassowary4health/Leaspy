import torch


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
        
    return index.sort()

def Sub_sampling(X,k):
    """
    Prend X le tensor (nb_visite,dim) et sélectionne k points bien espacé renvoyé dans un tensor (k,dim)

    testé approuvé


    """
    index=kmeans_plus_plus(X.numpy(), k)
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

def FiltreNanHomogène(XT,Y):
    """
    Prend en entrée XT (nb_patient,nb_visit_max,dim) et retourne X sous la forme (nb_visit,dim)

    Si un vecteur contient un Nan dans ses coordonnées on le retire

    testé approuvé

    """


    Select=((XT==XT).all(axis=2))*(Y==Y).all(axis=2)#fonctionne bien voir notebook test pour se convaincre
    
    return XT[Select],Y[Select]

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











def Matrix(X,Xgrand,meta_settings):
    """
    X est la donnée des points de controles un tensor de la forme (k,nb_dim) (k nombre de visite après subsampling), kernelname le nom du noyau à utiliser
    cette fonction renvoie la matrice K_X=(k(x_i,x_j)) i <nb_visit+1, j<k+1
    On a Xgrand (nb_visit,nb_dim) les points de controle sans subsampling
    """
    kernelname=meta_settings["kernelname"]
    sigma=meta_settings["sigma"]
    k=len(X)
    nb_visit=len(Xgrand)
    
    
    if kernelname=="RBF":#le calcul est fait sans approximations
        sigma=meta_settings["sigma"]
        

        PA1=Xgrand.repeat(k,1,1)
        PA2=X.repeat(nb_visit,1,1).permute(1,0,2)

        PA3=PA1-PA2

        K_X=torch.exp(-torch.norm(PA3,dim=2)**2/(2*sigma**2))
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


def solver(Matrice,Constante,meta_settings):
    """
    Prend en entrée les paramètres du problème quadratique (Matrice kxk + Constante vecteur de taille k)

    """
    raise ValueError(("not implemented"))
