# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:31:24 2020

@author: baptc
"""


import numpy as np
from scipy.stats import norm






def VaR_Normal_Distribution(rdt,var_quantile=0.01):
    """
    Static normal distribution
    """
    rdt=rdt.tail(250) # Le papier suggère de prendre 250 observations
    mu=np.mean(rdt)
    sigma=np.std(rdt)
    fi=norm.ppf(var_quantile)
    
    VaR=mu+sigma*fi
    
    return VaR


def VaR_HS(rdt,sample_size=250,var_quantile=0.01):
    """
    Historical simulation
    """
    mu=np.mean(rdt)
    sigma=np.std(rdt)
    sample=norm.rvs(loc=mu,scale=sigma, size=sample_size)
    sample=np.sort(sample)
    
    VaR=sample[int(sample_size*var_quantile)]
    
    return VaR


def VaR_Weighted_HS(rdt,sample_size=250,var_quantile=0.01):
    """
    Weighted historical simulation
    """
    mu=np.mean(rdt)
    sigma=np.std(rdt)
    rd=np.zeros((250,2)) # Le papier suggère de prendre 250 observations
    
    # on réalise une simulation de paramètres mu et sigma
    # on pondère les rendements en accordant un poids plus
    # important aux rendements les plus récents
    for i in range(0,250,1):
        rd[i,0]=norm.rvs(loc=mu,scale=sigma, size=1)
        rd[i,1]=(0.99**(i-1)*(1-0.99))/(1-0.99**250)


    rd=rd[rd[:,0].argsort()]

    seuil=0
    compteur=0
    while seuil<var_quantile:
        seuil=seuil+rd[compteur,1]
        VaR=rd[compteur,0]

    return VaR


def VaR_Riskmetrics(rdt,var_quantile=0.01):
    """
    RiskMetrics
    """

    sigma_t=np.var(rdt)
    rdt_t=float(rdt.tail(1))
    fi=norm.ppf(var_quantile)
    
    
    sigma_t1=0.06*(rdt_t**2)+0.94*sigma_t
    VaR=np.std(sigma_t1)*fi
   
    return VaR