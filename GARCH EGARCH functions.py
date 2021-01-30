#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from arch import arch_model
import statistics as stat
from scipy.stats import norm
from scipy.stats import t


#1 Modèle GARCH(1,1) avec résidus suivant une loi normale
#r étant le vecteur des rendement de chaque actif
 
def Garch_Normal(r,alpha):
    
    model=arch_model(r,mean="AR",lags=1,vol='GARCH',p=1,o=0,q=1,dist='Normal')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred=fit_model.forecast(horizon=1);
    var_1day=pred.variance.values[-1, :]
    VaR= -(norm.ppf(1-alpha)*np.sqrt(var_1day))
    
    return VaR

#2 Modèle GARCH(1,1) avec résidus suivant une loi de Student
#r étant le vecteur des rendement de chaque actif
def Garch_Student(r,alpha):

    #calcul de la moyenne
    rescale=False
    model=arch_model(r,mean="AR",lags=1,vol='GARCH',p=1,o=0,q=1,dist='studentst')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred=fit_model.forecast(horizon=1)
    var_1day=pred.variance.values[-1, :]
    VaR= -(t.ppf(1-alpha,df=10)*np.sqrt(var_1day))
    return VaR

#3 Modèle GARCH(1,1) avec utilisation de la distribution des résidus standardiés afin d'estimer les quantiles
#r étant le vecteur des rendement de chaque actif
def GARCH_FHS(r,alpha):

   
    model=arch_model(r,mean="AR",lags=1,vol='GARCH',p=1,o=0,q=1,dist='Normal')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    #pred=fit_model.forecast(start='1996-1-1',horizon=1)
    pred=fit_model.forecast(horizon=1)
    cond_mean = pred.mean.values[-1:]
    cond_var = pred.variance.values[-1:]
    #calcul des résidus standardisés
    std_rets = r / fit_model.conditional_volatility
    std_rets = std_rets.dropna()
    q = std_rets.quantile(alpha)
    VAR = - np.sqrt(cond_var) * q
    return VAR 

#1 Modèle EGARCH(1,1) avec résidus suivant une loi normale
#r étant le vecteur des rendement de chaque actif
def EGarch_Normal(r,alpha):
   

    model=arch_model(r,mean="AR",lags=1,vol='EGARCH',p=1,o=0,q=1,dist='Normal')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred=fit_model.forecast(horizon=1);
    var_1day=pred.variance.values[-1, :]
    VaR= -(norm.ppf(1-alpha)*np.sqrt(var_1day))
    return VaR
    
    
 #2 Modèle EGARCH(1,1) avec résidus suivant une loi de student
#r étant le vecteur des rendement de chaque actif    
def EGarch_Student(r,alpha):
    
    rescale=False
    model=arch_model(r,mean="AR",lags=1,vol='EGARCH',p=1,o=0,q=1,dist='studentst')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred=fit_model.forecast(horizon=1);
    var_1day=pred.variance.values[-1, :]
    VaR= -(t.ppf(1-alpha,df=10)*np.sqrt(var_1day))
    return VaR

#3 Modèle EGARCH(1,1) avec utilisation de la distribution des résidus standardiés afin d'estimer les quantiles
#r étant le vecteur des rendement de chaque actif
def EGarch_FHS(r,alpha):
    
    model=arch_model(r,mean="AR",lags=1,vol='EGARCH',p=1,o=0,q=1,dist='Normal')
    fit_model=model.fit()
    #Faire la prédiction à 1 jour de la variance
    pred=fit_model.forecast(horizon=1)
    cond_mean = pred.mean.values[-1:]
    cond_var = pred.variance.values[-1:]
    #calcul des résidus standardisés
    std_rets = r / fit_model.conditional_volatility
    std_rets = std_rets.dropna()
    #Calcul des quantiles des résidus standardisés
    q = std_rets.quantile(alpha)
    VAR = - np.sqrt(cond_var) * q
    return VAR 




# In[2]:


#Application
#récupération des données

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime


def get_rendements(stocks,début,fin): #date=yyyy-mm-dd en string
    prix=pdr.get_data_yahoo(stocks,start=début,end=fin)['Close'] 
    returns=prix.pct_change()
    logrdt=np.log(1+returns)*100
    logrdt=logrdt.iloc[1:]
    returns=returns.iloc[1:]
    prix=prix.iloc[1:]
    return  returns, prix, logrdt

stocks=['AAPL','AXP','BA', 'CAT', 'CSCO', 'CVX','DD','DIS','GE','HD','HPQ','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','RTX,''T','TRV','UNH','VZ','WMT','XOM']
début='1995-12-29'
fin='2014-12-31'

output=get_rendements(stocks, début, fin)
rendements=output[0]
prix=output[1]
rdt_log=output[2]


# In[4]:


#Application

import matplotlib.pyplot as plt
import numpy as np
from pandas import*
from os.path import dirname
import pathlib

#création de la matrice des VaR pour chaque actif
VaR_EGARCH_Normal=np.zeros((len(rendements), len(stocks)))
#fenetre roulante de 1000
wd_size=1000

j = 0
#boucle sur les 30 actifs
for stock in stocks:
    rdt = rdt_log.iloc[:, j]  # par exemple pour APPLE
   
    #boucle sur les périodes de temps
    for i in range(wd_size, len(rdt_log)):
        VaR_EGARCH_Normal[i,j]=EGarch_Normal(rdt[i-wd_size:i],alpha=0.01)
    print(t1-t0)
    j += 1

#transformation en dataframe et stocjage en format csv
VaR_EGARCH_Normal=pandas.DataFrame(VaR_EGARCH_Normal,index=rendements.index)

VaR_EGARCH_Normal.to_csv (os.path.dirname(__file__)+"\\"+"EGARCH Normal"+".csv")

