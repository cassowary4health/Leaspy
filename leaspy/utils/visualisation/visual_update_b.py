import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from leaspy import Leaspy, Data, AlgorithmSettings, Plotter, Dataset, IndividualParameters





def plot_average_update(model,ax,modelref=None,name=None,time=[60,90]):
    """
    Parameters: model, linear_b model already calibrated with update_b
        ax, (dimension) subplots

    """
    timepoints = np.linspace(time[0], time[1], 100)
    if modelref is not None:
        number_of_sources = modelref.model.random_variable_informations()["sources"]["shape"][0]
        mean_xiref = modelref.model.parameters['xi_mean'].numpy()
        mean_tauref = modelref.model.parameters['tau_mean'].numpy()
        mean_sourceref = modelref.model.parameters['sources_mean'].numpy().tolist()
        mean_sourcesref = [mean_sourceref]*number_of_sources
        
       
        
        average_parametersref = {'xi': mean_xiref,'tau': mean_tauref,'sources': mean_sourcesref}
        ip_averageref = IndividualParameters()
        ip_averageref.add_individual_parameters('average', average_parametersref)
        valuesref = modelref.estimate({'average': timepoints}, ip_averageref)
        for i in range(model.model.dimension):
            ax[i].plot(timepoints, valuesref['average'].T[i], linewidth=3,label="reference",c="red")


    # —— Get the average individual parameters
    d=model.model.dimension
    
    Lparam=model.model.saveParam
    LB=model.model.saveB.copy()
    print(len(LB))
    cm = plt.get_cmap('winter')

    for i in range(d):
        
        number_of_sources = model.model.random_variable_informations()["sources"]["shape"][0]
        
        for j in range(len(LB)):
            if i==0:
                print("norme W")
                print(torch.norm(torch.tensor(LB[j][0]),dim=0))
            
            if type(Lparam[j]['xi_mean']) is torch.Tensor:
                mean_xi = Lparam[j]['xi_mean'].numpy()
                mean_tau = Lparam[j]['tau_mean'].numpy()
                mean_source = Lparam[j]['sources_mean'].numpy().tolist()
            else:
                mean_xi = Lparam[j]['xi_mean']
                mean_tau = Lparam[j]['tau_mean']
                mean_source = Lparam[j]['sources_mean']
            mean_sources = [mean_source]*number_of_sources
            model.model.saveB=LB[:j]
            model.model.reconstructionB()

# —— Store the average individual parameters in a dedicated object
            average_parameters = {'xi': mean_xi,'tau': mean_tau,'sources': mean_sources}
            ip_average = IndividualParameters()
            ip_average.add_individual_parameters('average', average_parameters)
            values = model.estimate({'average': timepoints}, ip_average)
            
            if j==0:
                ax[i].plot(timepoints, values['average'].T[i], linewidth=1,label="init ",c=cm(1.*j/(len(LB)-1)))
            else:
                ax[i].plot(timepoints, values['average'].T[i], linewidth=1,label="comp "+str(j),c=cm(1.*j/(len(LB)-1)))
        ax[i].set_xlabel("time")
        ax[i].set_ylabel("dim"+str(i))
        
    model.model.saveB=LB



def plot_variability(model, tps,nb_vari,fen):
    params={}
    

    nb_s=model.model.source_dimension
    print(nb_s)
    dimension=model.model.dimension
    fig1, ax1 = plt.subplots(dimension, nb_s, figsize=(4*dimension,16*nb_s))
    for i in range(nb_s):
        SS=torch.zeros(nb_vari,nb_s)
        SS[:,i]=torch.linspace(-fen,fen,nb_vari)
        
        t = tps

        
        for j in range(dimension):
            for k in range(nb_vari):
        
                mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                mean_par["sources"] = SS[k]
                trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
                
                if len(ax1.shape)>1:
                    ax1[j,i].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]))
                    ax1[j,i].set_xlabel("time")
                    ax1[j,i].set_ylabel("dim"+str(j))
                else:
                    ax1[j].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]))
                    ax1[j].set_xlabel("time")
                    ax1[j].set_ylabel("dim"+str(j))
    plt.legend()
    plt.show()