#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from arch import arch_model
import statistics as stat
from scipy.stats import norm
from scipy.stats import t

#1 Modèle GARCH(1,1) avec résidus suivant une loi normale
#r étant le vecteur des rendement du ptf équipondéré
def Garch_Normal(r,alpha):
   
    
    p=1
    o=0
    q=1
    #calcul de la moyenne
    r_mean=stat.mean(r)
    #calcul des rendements corrigés de la moyenne
    r_corr= r-mean(r)
    model=arch_model(r_corr,mean="AR",lags=1,vol='GARCH',p=1,o=0,q=1,dist='Normal')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred_=fit_model.forecast(horizon=1);
    var_1day=pred.variance.values[-1, :]
    VaR= (norm.ppf(1-alpha)*np.sqrt(var_1day))
    
def Garch_Student(r,alpha):

    p=1
    o=0
    q=1
    #calcul de la moyenne
    r_mean=stat.mean(r)
    #calcul des rendements corrigés de la moyenne
    r_corr= r-mean(r)
    model=arch_model(r_corr,mean="AR",lags=1,vol='GARCH',p=1,o=1,q=1,dist='studentst')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred_=fit_model.forecast(horizon=1);
    var_1day=pred.variance.values[-1, :]
    VaR= (t.ppf(1-alpha,df=10)*np.sqrt(var))
    
def EGarch_Normal(r,alpha):
   
    
    p=1
    o=0
    q=1
    #calcul de la moyenne
    r_mean=stat.mean(r)
    #calcul des rendements corrigés de la moyenne
    r_corr= r-mean(r)
    model=arch_model(r_corr,mean="AR",lags=1,vol='EGARCH',p=1,o=0,q=1,dist='Normal')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred_=fit_model.forecast(horizon=1);
    var_1day=pred.variance.values[-1, :]
    VaR= (norm.ppf(1-alpha)*np.sqrt(var_1day))
    
    
    
def EGarch_Student(r,alpha):
    
    p=1
    o=0
    q=1
    #calcul de la moyenne
    r_mean=stat.mean(r)
    #calcul des rendements corrigés de la moyenne
    r_corr= r-mean(r)
    model=arch_model(r_corr,mean="AR",lags=1,vol='EGARCH',p=1,o=0,q=1,dist='studentst')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred_=fit_model.forecast(horizon=1);
    var_1day=pred.variance.values[-1, :]
    VaR= (t.ppf(1-alpha)*np.sqrt(var_1day))




# In[21]:


from scipy.stats import t
t.ppf(0.95,df=10)

