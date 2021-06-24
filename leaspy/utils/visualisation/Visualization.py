import os
import pandas as pd
import sys
import json
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


user_path=os.getcwd()

from leaspy import Leaspy, AlgorithmSettings,IndividualParameters, Data, Dataset

current_directory = user_path
input_directory = os.path.join(current_directory, 'data')
output_directory = user_path


def plot_average_update(model,ax,modelref=None,name=None,time=[60,90],dim=None,features=None):
    """
    Parameters: model, linear_b model already calibrated with update_b
        ax, (dimension) subplots

    """
    if features is not None:
        model.model.features=features
    features=model.model.features
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
    cm = plt.get_cmap('autumn')
    if dim is None:
        for i in range(d):
        
            number_of_sources = model.model.random_variable_informations()["sources"]["shape"][0]
        
            for j in range(len(LB)):
                if i==0:
                    print("norme of the kernel weights at step "+str(j))
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
                D=model.model.saveParam[j]
                print("velocities v0 at step "+str(j))
                print(D["v0"])
                model.model.load_parameters(D)
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
            ax[i].set_xlabel("Alzeihmer age")
            ax[i].set_ylabel(features[i])
        
        model.model.saveB=LB
    else:
        number_of_sources = model.model.random_variable_informations()["sources"]["shape"][0]
        i=dim
        for j in range(len(LB)):
            if i==0:
                print("norme of the kernel weights at step "+str(j))
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
            D=model.model.saveParam[j]
            print("velocities v0 at step "+str(j))
            print(D["v0"])
            model.model.load_parameters(D)
            model.model.saveB=LB[:j]
            model.model.reconstructionB()

# —— Store the average individual parameters in a dedicated object
            average_parameters = {'xi': mean_xi,'tau': mean_tau,'sources': mean_sources}
            ip_average = IndividualParameters()
            ip_average.add_individual_parameters('average', average_parameters)
            values = model.estimate({'average': timepoints}, ip_average)

            if j==0:
                ax.plot(timepoints, values['average'].T[i], linewidth=2,label="step 1 ",c=cm(1-1.*j/(len(LB)-1)))
            else:
                if j==len(LB)-1:
                    ax.plot(timepoints, values['average'].T[i], linewidth=1.5,label="step "+str(j+1),c=cm(1-1.*j/(len(LB)-1)))
                else:
                    ax.plot(timepoints, values['average'].T[i], linewidth=1,label="step "+str(j+1),c=cm(1-1.*j/(len(LB)-1)))
            plt.ylim(0, 1)
            ax.set_xlabel("Alzeihmer age",fontsize ="large")
            ax.set_ylabel(features[i],fontsize ="large")




def plot_variability(model, tps,nb_vari,fen,dim=None):
    params={}
    
    print(model)
    nb_s=model.model.source_dimension
    print(nb_s)
    features=model.model.features
    dimension=model.model.dimension
    cm = plt.get_cmap('winter')
    if dim is None:

        fig1, ax1 = plt.subplots(dimension, nb_s, figsize=(4*dimension,16*nb_s))
        for i in range(nb_s):
            SS=torch.zeros(nb_vari,nb_s)
            SS[:,i]=torch.linspace(0,fen,nb_vari)
        
            t = tps

        
            for j in range(dimension):
                for k in range(nb_vari):
        
                    mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                    mean_par["sources"] = SS[k]
                    trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
                
                    if len(ax1.shape)>1:
                        if k==0:
                            ax1[j,i].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]),color="red")
                        else:
                            ax1[j,i].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]),c=cm(1.*k/(nb_vari-1)))
                            mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                            mean_par["sources"] = -SS[k]
                            trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
                            ax1[j,i].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]),c=cm(1.*k/(nb_vari-1)))
                        ax1[j,i].set_xlabel("Alzeihmer age")
                        ax1[j,i].set_ylim([0,1])
                        ax1[j,i].set_ylabel(features[j])
                    else:
                        if k==0:
                            ax1[j].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]),color="red")
                        else:
                            ax1[j].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]),c=cm(1.*k/(nb_vari-1)))
                            mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                            mean_par["sources"] = -SS[k]
                            trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
                            ax1[j].plot(t, trajectory[..., j],label="source "+str(i)+": {:.2f}".format(SS[k,i]),c=cm(1.*k/(nb_vari-1)))
                        ax1[j].set_xlabel("Alzeihmer age",fontsize ="medium")
                        ax1[j].set_ylabel(features[j])
    else:
        fig1, ax1 = plt.subplots( nb_s,1, figsize=(10,10))
        plt.ylim(0, 1)
        for i in range(nb_s):
            SS=torch.zeros(nb_vari,nb_s)
            SS[:,i]=torch.linspace(0,fen,nb_vari)
        
            t = tps
            j=dim
            for k in range(nb_vari):
        
                mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                mean_par["sources"] = SS[k]
                trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
                
                if len(ax1)>1:
                    
                    if k==0:
                            
                            ax1[i].plot(t, trajectory[..., j],color="red")
                    else:
                        
                        ax1[i].plot(t, trajectory[..., j],c=cm(1.*k/(nb_vari-1)))
                        mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                        mean_par["sources"] = -SS[k]
                        trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
                        ax1[i].plot(t, trajectory[..., j],c=cm(1.*k/(nb_vari-1)))
                    ax1[i].set_ylim([0,1])
                    ax1[i].set_xlabel("Alzeihmer age",fontsize ="large")
                    ax1[i].set_ylabel(features[j],fontsize ="large")
                else:
                    ax1.plot(t, trajectory[..., j])
                    ax1.set_xlabel("Alzeihmer age")
                    ax1.set_ylabel(features[j],fontsize ="medium")


        #plt.legend()
        plt.show()

def plot_corne(model, tps,nb_vari,fen,dim1,dim2,nb_source=2):
    params={}
    
    print(model)
    nb_s=nb_source
    print(nb_s)
    tpmin=min(tps)
    tpleft=torch.linspace(0,tpmin,1000)
    tpmax=max(tps)
    tpright=torch.linspace(tpmax,130,1000)
    features=model.model.features
    dimension=model.model.dimension
    cm = plt.get_cmap('winter')
    fig1, ax1 = plt.subplots( nb_s,1, figsize=(5,10))
    for i in range(nb_s):
        SS=torch.zeros(nb_vari,nb_s)
        SS[:,i]=torch.linspace(0,fen,nb_vari)
        
        t = tps
        tl=tpleft
        tr=tpright
        
        for k in range(nb_vari):
        
            mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
            mean_par["sources"] = SS[k]
            trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
            if k==0:
                            
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],color="red")
            else:
                        
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],c=cm(1.*k/(nb_vari-1)))
                mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                mean_par["sources"] = -SS[k]
                trajectory = model.model.compute_individual_tensorized(t, mean_par).squeeze(0)
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],c=cm(1.*k/(nb_vari-1)))
            mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
            mean_par["sources"] = SS[k]
            trajectory = model.model.compute_individual_tensorized(tl, mean_par).squeeze(0)
            if k==0:
                            
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],color="red",alpha=0.2)
            else:
                        
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],c=cm(1.*k/(nb_vari-1)),alpha=0.2)
                mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                mean_par["sources"] = -SS[k]
                trajectory = model.model.compute_individual_tensorized(tl, mean_par).squeeze(0)
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],c=cm(1.*k/(nb_vari-1)),alpha=0.2)
            mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
            mean_par["sources"] = SS[k]
            trajectory = model.model.compute_individual_tensorized(tr, mean_par).squeeze(0)
            if k==0:
                            
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],color="red",alpha=0.2)
            else:
                        
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],c=cm(1.*k/(nb_vari-1)),alpha=0.2)
                mean_par={"tau":model.model.parameters["tau_mean"],"xi":model.model.parameters["xi_mean"]}
                mean_par["sources"] = -SS[k]
                trajectory = model.model.compute_individual_tensorized(tr, mean_par).squeeze(0)
                ax1[i].plot(trajectory[..., dim1], trajectory[..., dim2],c=cm(1.*k/(nb_vari-1)),alpha=0.2)
            ax1[i].set_ylim([0,1])
            ax1[i].set_xlabel(features[dim1],fontsize="large")
            ax1[i].set_ylabel(features[dim2],fontsize="large")
    plt.legend()
    plt.show()

def variability_corne(path_exp,tps,nb_vari,fen,dim1,dim2,features=None,nb_source=2):
    path_output_calibrate = os.path.join(path_exp, "calibrate")
    i="2"
    name_save="pred_individual_parameters_"+i+".json"
    path_output_personalize = os.path.join(path_exp, "personalize","pred")
    path_output_personalize_1 = os.path.join(path_output_personalize,name_save)
           
    path=os.path.join(path_output_calibrate,"fold_"+i,"model_parameters.json")
    Model=Leaspy.load(path)
    print(torch.exp(Model.model.parameters["Param"][Model.model.dimension:]))
    if features is not None:
        Model.model.features=features
    
    plot_corne( Model, tps,nb_vari,fen,dim1,dim2,nb_source) 

def variability(path_exp,tps,nb_vari,fen,dim=None,features=None):
    path_output_calibrate = os.path.join(path_exp, "calibrate")
    i="2"
    name_save="pred_individual_parameters_"+i+".json"
    path_output_personalize = os.path.join(path_exp, "personalize","pred")
    path_output_personalize_1 = os.path.join(path_output_personalize,name_save)
           
    path=os.path.join(path_output_calibrate,"fold_"+i,"model_parameters.json")
    Model=Leaspy.load(path)
    if features is not None:
        Model.model.features=features
    print(Model.model.parameters["v0"])
    plot_variability( Model, tps,nb_vari,fen,dim)

def variabilityINIT_corne(path_exp,tps,nb_vari,fen,dim1,dim2,features=None):
    path_output_calibrate = os.path.join(path_exp, "calibrate")
    i="2"
    name_save="pred_individual_parameters_"+i+".json"
    path_output_personalize = os.path.join(path_exp, "personalize","pred")
    path_output_personalize_1 = os.path.join(path_output_personalize,name_save)
           
    path=os.path.join(path_output_calibrate,"fold_"+i,"model_parameters.json")
    Model=Leaspy.load(path)
    n_comp=0
    L=Model.model.saveB[:n_comp]
    D=Model.model.saveParam[n_comp]
    print(D["v0"])
    Model.model.load_parameters(D)
    Model.model.saveB=L
    Model.model.initBlink()
    Model.model.reconstructionB()
    if features is not None:
        Model.model.features=features
    plot_corne( Model, tps,nb_vari,fen,dim1,dim2)


def variabilityINIT(path_exp,tps,nb_vari,fen,dim=None,features=None):
    path_output_calibrate = os.path.join(path_exp, "calibrate")
    i="2"
    name_save="pred_individual_parameters_"+i+".json"
    path_output_personalize = os.path.join(path_exp, "personalize","pred")
    path_output_personalize_1 = os.path.join(path_output_personalize,name_save)
           
    path=os.path.join(path_output_calibrate,"fold_"+i,"model_parameters.json")
    Model=Leaspy.load(path)
    n_comp=0
    L=Model.model.saveB[:n_comp]
    D=Model.model.saveParam[n_comp]
    print(D["v0"])
    Model.model.load_parameters(D)
    Model.model.saveB=L
    Model.model.initBlink()
    Model.model.reconstructionB()
    if features is not None:
        Model.model.features=features
    plot_variability( Model, tps,nb_vari,fen,dim)