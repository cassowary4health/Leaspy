
from .abstract_regressor import AbstractRegressor
import numpy as np
from scipy import stats

class LogisticRegressor(AbstractRegressor):


    def __init__(self, **kwargs):
        super().__init__("logistic", **kwargs)
        self.parameters = None#mettre
        
        def init(T,Y):
            initi=np.zeros(2)
            tmean=T.mean()
            T1=T-tmean
            ymean=np.clip(Y.mean(),a_min=0.01,a_max=0.99)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(T1, Y)
            initi[1]=slope/(ymean*(1-ymean))
            initi[0]=1/ymean-1
            return T1,initi

        self.initializer=init
        self.threshold=3
    #implémenter contraintes ?
    @staticmethod
    def fonction(T):#à implémenter
        """renvoie la fonction associé à T, cette fonction prends en entrée un vecteur de la taille de la dimension
        des paramètres et renvoit un vecteur de la taille de T """
        def logistic(param):
            L=np.ones(len(T))/(1.+param[0]*np.exp(-param[1]*T))
            return L
        return logistic

    @staticmethod
    def gradient(T):#à implémenter
        """renvoie la fonction gradient associé à T, cette fonction renvoit un tableau de taill(len(T),len(parameters)) """
        def logistic_grad(param):
            L1=-np.exp(-param[1]*T)/(1.+param[0]*np.exp(-param[1]*T))**2
            L2=-param[0]*np.exp(-param[1]*T)*T/(1.+param[0]*np.exp(-param[1]*T))**2
            Cac=np.concatenate([L1.reshape((-1,1)),L2.reshape((-1,1))],axis=1)
            return Cac


        return logistic_grad


    
       
