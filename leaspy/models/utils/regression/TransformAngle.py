import numpy as np
import math
import torch

def vectoangle(s):
    """
    Parameters:
    -s an normalized vector of dimension d
    return:
    - theta the angles vector of dimension d-1

    """
    if len(s.shape)==1:
        d=len(s)
        L=torch.zeros(d-1)
        L[0]=torch.acos(s[0])
        S=1.0
        for i in range(1,d-1):
            S=torch.sin(L[i-1])*S
            L[i]=torch.acos(s[i]/S)
        return L
    elif len(s.shape)==2:
        d=s.shape[1]
        N=s.shape[0]
        L=torch.zeros(N,d-1)
        L[:,0]=torch.acos(s[:,0])
        S=torch.ones(N)
        for i in range(1,d-1):
            S=torch.sin(L[:,i-1])*S
            L[:,i]=torch.acos(s[:,i]/S)
        return L
    else:
        raise ValueError

def angletovec(theta):
    """
    Parameters:
    -s an normalized vector of dimension d
    - theta the angles vector of dimension d-1
    return:
    -s an normalized vector of dimension d
    """
    if len(theta.shape)==1:
        dminusone=len(theta)
        L=torch.zeros(dminusone+1)
        L[0]=torch.cos(theta[0])
        S=1.0
        
        for i in range(1,dminusone):
            S=torch.sin(theta[i-1])*S
            L[i]=torch.cos(theta[i])*S
        S=torch.sin(theta[dminusone-1])*S
        L[dminusone]=S
        return L
    elif len(theta.shape)==2:
        N=theta.shape[0]
        dminusone=theta.shape[1]
        L=torch.zeros(N,dminusone+1)
        L[:,0]=torch.cos(theta[:,0])
        S=torch.ones(N)
        
        for i in range(1,dminusone):
            S=torch.sin(theta[:,i-1])*S
            L[:,i]=torch.cos(theta[:,i])*S
        S=torch.sin(theta[:,dminusone-1])*S
        L[:,dminusone]=S
        return L
    else:
        raise ValueError

def rotation(w,thetaplus):
    if len(w.shape)==1:
        norm=torch.norm(w)
        wnorm=w.clone()
        wnorm=w/norm
        repangle=vectoangle(wnorm)
        rep=repangle+thetaplus
        wnew=angletovec(rep)*norm
        return wnew
    elif len(w.shape)==2:
        norm=torch.norm(w,dim=1)
        wnorm=w.clone()
        wnorm[norm>10**(-5)]=w[norm>10**(-5)]/norm.unsqueeze(-1)[norm>10**(-5)]
    
        repangle=vectoangle(wnorm)
        
        rep=repangle+thetaplus
        wnew=angletovec(rep)*norm.unsqueeze(-1)
        
        return wnew
    else:
        raise ValueError

