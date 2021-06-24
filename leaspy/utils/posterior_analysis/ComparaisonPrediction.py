import os
import pandas as pd
import sys
import json
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import random
import plotly.express as px
from joblib import Parallel, delayed



from leaspy import Leaspy, AlgorithmSettings,IndividualParameters, Data, Dataset




# Algo parameters
personalize_algorithm = "scipy_minimize"
n_iter_personalize = 100
seed = 0
algo_settings_personalize = AlgorithmSettings(personalize_algorithm, seed=seed, n_iter=n_iter_personalize)


def decomp(L,mask,dataset,res):
    Z=[]
    TT=[]
    Dim=[]
    
    
    
    a,b,c=mask.shape
    Val=[]
    Pred=[]
    for j in range(c):
        Val.append([])
        Pred.append([])
    
    for i in range(a):
        e=L[i]
        t=0
        for j in range(b):
            
            visi=False
            for k in range(c):
                if mask[i,j,k]==1:
                    Z=Z+[int(e[-1])]
                    TT=TT+[t]
                    visi=True
                    Dim.append(k)
                    for g in range(c):
                        Val[g].append(dataset[i,j,g].item())
                        Pred[g].append(res[i,j,g].item())
            if visi:
                t=t+1
    return Z,TT,Dim,Val,Pred

def all_res(model,res,dataset):
    D={}
    Z,TT,Dim,Val,Pred=decomp(dataset.indices,dataset.mask.float(),dataset.values,res)
    D["indices"]=Z
    D["nb_step"]=TT
    nb_ob=torch.tensor(dataset.nb_observations_per_individuals).unsqueeze(-1)
    D["AE"]=torch.abs(dataset.mask.float() * (res - dataset.values))
    D["AE"]=D["AE"][dataset.mask.bool()]
    D["dim"]=Dim
    D["Val"]=Val
    D["Pred"]=Pred
    return D
def compute_predictions(leaspy_data, model, results,name,sources=True):

    
    dataset = Dataset(leaspy_data)
    
    param_ind = results._individual_parameters
    
    params = {}
    keys = ['tau', 'xi']
    if sources:
        keys.append('sources')

    if name=="logistic_asymp_delay":
        keys.append('sources_asymp')
    for key in keys:
        values = torch.tensor([ind[key] for idx, ind in param_ind.items()])
        if key in ['tau', 'xi']:
            params[key] = values.unsqueeze(1)
        else:
            params[key] = values
   
    trajectories = model.compute_individual_tensorized(dataset.timepoints, params, None)

    return dataset, trajectories

def Split_visits_imputation(df,nfit=2,threshold=3):
    pred_data_X = []
    pred_data_Y = []
    random.seed(0)
    for idx in pd.unique(df.index.get_level_values(0)):
        indiv = df.loc[idx]
        N = indiv.shape[0]
        LN=set(np.arange(N))
        if N >= threshold:
            
            for k in range(1,N-nfit+1):
                
                indiv.reset_index(inplace=True)
                    
                indiv['ID'] = str(idx)+"_"+str(k)
                indiv.set_index(['ID', 'TIME'], inplace=True)
                A=random.sample(range(N),k)
                B=LN-set(A)
                x = indiv.iloc[list(B)]
                y = indiv.iloc[A]
                pred_data_X.append(x)
                pred_data_Y.append(y)
                
    pred_data_X = pd.concat(pred_data_X)
    pred_data_Y = pd.concat(pred_data_Y)
    return pred_data_X,pred_data_Y

def Split_visits2(df,threshold=3):
    pred_data_X = []
    pred_data_Y = []
    for idx in pd.unique(df.index.get_level_values(0)):
        indiv = df.loc[idx]
        N = indiv.shape[0]
        if N >= threshold+1:
            c = 0
            for k in range(threshold, N-1):
                
                indiv.reset_index(inplace=True)
                    
                indiv['ID'] = str(idx)+"_"+str(c)
                indiv.set_index(['ID', 'TIME'], inplace=True)
                x = indiv.iloc[:k]
                y = indiv.iloc[N-1:]
                pred_data_X.append(x)
                pred_data_Y.append(y)
                c += 1
    pred_data_X = pd.concat(pred_data_X)
    pred_data_Y = pd.concat(pred_data_Y)
    return pred_data_X,pred_data_Y


def PersoSpecial(df,path_exp,name,algo_settings_personalize,Score,kernel_sec=False,n_comp=0,fold_max=5,predic=True,th=3,imputation=False,nfit=2):
    path_output_calibrate = os.path.join(path_exp, "calibrate")
    path_to_indices=os.path.join(path_output_calibrate,"resampling_indices.json")
    with open(path_to_indices) as fp:
            split_indices = json.load(fp)
    def loadmodel(i):
        path=os.path.join(path_output_calibrate,"fold_"+i,"model_parameters.json")
        Model=Leaspy.load(path)
        return Model
    Test={}
    path_output_personalize = os.path.join(path_exp, "personalize",name)
    if not os.path.exists(path_output_personalize):
        os.makedirs(path_output_personalize)
    
    for j in range(fold_max):
        i=str(j)
        train,test=split_indices[i]
        
        Mod=loadmodel(i)

        name_save="pred_individual_parameters_"+i+".json"
        if kernel_sec:#Enable to select the model associated to a given step n_comp
            print(Mod.model.initB)
            L=Mod.model.saveB[:n_comp]
            D=Mod.model.saveParam[n_comp]
            Mod.model.load_parameters(D)
            Mod.model.saveB=L
            Mod.model.initBlink()
            Mod.model.reconstructionB()
            name_save="pred_ncomp_"+str(n_comp)+"_individual_parameters_"+i+".json"
        

        df_split_test = df.loc[test].reset_index()
        df_split_test.set_index(['ID','TIME'], inplace=True)
        indices = [idx for idx in df_split_test.index.unique('ID') if df_split_test.loc[idx].shape[0] >= th+1]
        df_split_test = df_split_test[df_split_test.index.get_level_values(0).isin(indices)] 
        if predic==True:
            if imputation:
                pred_data_X,pred_data_Y=Split_visits_imputation(df_split_test,nfit=nfit,threshold=th)
            else:
                pred_data_X,pred_data_Y=Split_visits2(df_split_test,threshold=th)
            X = Data.from_dataframe(pred_data_X)
        else:
            X = Data.from_dataframe(df_split_test)
        path_output_personalize_1 = os.path.join(path_output_personalize,name_save)
        if not os.path.exists(path_output_personalize_1):
            results=Mod.personalize(X,algo_settings_personalize)
            results.save(path_output_personalize_1)
        else:
            results=IndividualParameters.load(path_output_personalize_1)   
    
        
        if predic==True:
            Y = Data.from_dataframe(pred_data_Y)
        if predic==True:
            dataset, pred = compute_predictions(Y, Mod.model, results,Mod.model.name)
            score=Score(Mod,pred,dataset)
        else:
            dataset, pred = compute_predictions(X, Mod.model, results,Mod.model.name)
            score=Score(Mod,pred,dataset)
        if len(Test)==0:
            for e in score.keys():
                Test[e]=[]

        for e in score.keys():
            Test[e].append(score[e])
    return Test


def PredTodataframe(Test,modelname="Nan"):
    Z=Test["AE"]
    B=Test["indices"]
    C=Test["nb_step"]
    D=Test["dim"]
    Val=Test["Val"]
    Pred=Test["Pred"]
  
    L=[]
    for i in range(len(Z)):
        
        
        dg=pd.DataFrame()
        dg["error"]=Z[i]
        
        dg["cat"]=B[i]
        dg["step"]=C[i]
        dg["dim"]=D[i]
        
        
        for k in range(len(Val[0])):
            
            dg['val_'+str(k)]=Val[i][k]
            dg['pred_'+str(k)]=Pred[i][k]
        dg["fold"]=[i]*len(Z[i])
        dg["model"]=[modelname]*len(Z[i])
        
        L.append(dg.copy())
    dh=pd.concat(L)
    return dh


import scipy.stats as stat



def compvisu(dh1Tot,dh2Tot,inspect="step",filtre="dim",n=1):
    dh1=dh1Tot[dh1Tot["cat"]==n]
    dh2=dh2Tot[dh2Tot["cat"]==n]
    A=set(dh1[filtre])
    n=len(A)
    #fig, ax = plt.subplots(n, 1, figsize=(4*n,16))
    for e in A:
        New=pd.DataFrame()
        New[dh1["model"].values[0]]=dh1[dh1[filtre]==e]["error"].values
        New[dh2["model"].values[0]]=dh2[dh2[filtre]==e]["error"].values
        New[inspect]=dh2[dh2[filtre]==e][inspect].values
        fig=px.scatter(New,x=dh1["model"].values[0],y=dh2["model"].values[0],color=inspect)
        fig.show()

def MAE_comparison(dh1Tot,dh2Tot,n=1,features=None):
    dh1=dh1Tot[dh1Tot["cat"]==n-1]
    dh2=dh2Tot[dh2Tot["cat"]==n-1]
    
    A=set(dh1["dim"])
    for e in A:
        if features is not None:
            print(features[e])
        else:
            print("dim "+str(e))
        print("MAE")
        print("GB : {:.3f}, DCM: {:.3f}".format(np.mean(dh1[dh1["dim"]==e]["error"].values),np.mean(dh2[dh1["dim"]==e]["error"].values)))
        print("quantile 0.90")
        print("GB : {:.3f}, DCM: {:.3f}".format(np.quantile(dh1[dh1["dim"]==e]["error"].values,0.90),np.quantile(dh2[dh2["dim"]==e]["error"].values,0.90)))
        print("Wilcoxon p-value")
        print(stat.wilcoxon(dh1[dh1["dim"]==e]["error"],dh2[dh2["dim"]==e]["error"]))