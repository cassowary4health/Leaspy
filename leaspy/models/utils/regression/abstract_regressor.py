import math
from abc import ABC, abstractmethod

from scipy.optimize import minimize
import torch
import numpy as np


class AbstractRegressor(ABC):
    """
    Contains the common attributes & methods of the different regressor.

    Attributes
    ----------
    is_initialized : bool
        Indicates if the model is initialized
    name : str
        The model's name
    features : list[str]
        Names of the model features
    parameters : dict
        Contains the model's parameters
    loss : str
        The loss to optimize (``'MSE'``, ``'MSE_diag_noise'`` or ``'crossentropy'``)
    
    """

    def __init__(self, name: str):
        self.is_initialized: bool = False
        self.name = name
        self.initializer=None
        self.threshold=None
        
        self.parameters = None
        self.loss: str = 'MSE'  # default value, changes when a fit / personalize algo is called, TODO: change to MSE_diag_noise ?
    

    @abstractmethod
    def fonction(self,T):
        """renvoie la fonction associé à T, cette fonction prends en entrée un vecteur de la taille de la dimension
        des paramètres et renvoit un vecteur de la taille de T """
        raise NotImplementedError
    
    @abstractmethod
    def gradient(self,T):
        """renvoie la fonction gradient associé à T, cette fonction renvoit un tableau de taill(len(T),len(parameters)) """
        raise NotImplementedError
    
    def fit(self,T,Y,init=None,method="Powell"):
        T1,x0=self.initializer(T,Y)
        if init is not None:
            x0=init
        def func_tomin(param):
            func=self.fonction(T1)
            return np.linalg.norm(func(param)-Y,2)**2/2
        def jac_tomin(param):
            func=self.fonction(T1)
            grad=self.gradient(T1)
            Z=func(param)-Y
            J=np.dot(Z,grad(param))
            return J
        if method=="BFGS":
            OptiResult=minimize(fun=func_tomin,x0=x0,method=method,jac=jac_tomin)
        else:
            OptiResult=minimize(fun=func_tomin,x0=x0,method=method,bounds=[(0,1000),(0,1000)])
       
        self.parameters=OptiResult.x
       
        return OptiResult.x,OptiResult.success,T1

    def ultra_fit(self,data):
        """
        Prends en entrées un objet dataset leaspy et renvoie les apramètres associés au trajectoire des patients

        """
        df = data.to_pandas()
        df.set_index(["ID", "TIME"], inplace=True)
        Param={}
        Time={}
        for idx in data.indices:
            param_dim_patients = []
            
            for dim in range(data.dimension):
                df_patient = df.loc[idx]
                df_patient_dim = df_patient.iloc[:, dim].dropna()
                x = df_patient_dim.index.get_level_values('TIME').values
                y = df_patient_dim.values
                
                if len(x) < self.threshold:
                    continue

                para, success,tt = self.fit(x, y)

                if success:
                    
                    param_dim_patients.append(para.tolist())
            if len(param_dim_patients)==4:
                Param[idx]=np.array(param_dim_patients)
                Time[idx]=tt
        

        return Param,Time#(n_patient,n_dim,n_dim_parameter)

    def loss(Y,T):
        param=self.parameters
        if self.loss=="MSE":
            func=self.fonction(T)
            return np.linalg.norm(func(param)-Y,2)**2/len(T)
    
    def predict(self,T):
        func=self.fonction(T)
        raise func(self.parameters)