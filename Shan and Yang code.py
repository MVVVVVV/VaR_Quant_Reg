#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime


#Sequential relative performance approach:


def Tick_Loss(u,alpha):
    """
    Fonction de perte aymétrique et par morceaux de Giacomini et Komunjer (2005)
    """ 
    
    if u<=0:
        y=1
    else:
        y=0
        
    return (alpha-y)*u  


def seq_perf(forecasts, rdt,alpha=0.01):
    
    M=forecasts.shape[1]
    T=forecasts.shape[0]
    error=np.zeros((T,M))
    beta=np.zeros((T,M))
    num=np.zeros((T,M))
    denom=np.zeros(T)
    
    #on fixe les betas initiaux pour chaque modèle:
    beta[0,:]=1/M
  
    # erreur de prévison:
    for j in range(0,M):
        
        forecast=forecasts.iloc[:,j]
        
                   
        for i in range(1,T):
            error[i,j]=Tick_Loss(rdt[i]-forecast.iloc[i],alpha)       
            num[i,j]=beta[i-1,j]*np.exp(-1*error[i,j])
            denom[i]=np.sum(num[i,:],axis=0)
            beta[i,j]=num[i,j]/denom[i]
    
    
    
    return beta


    

