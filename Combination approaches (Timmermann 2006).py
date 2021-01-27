# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:03:28 2021

@author: baptc
"""

import numpy as np
import numpy.random as npr
import math




def Trimmed_Mean(M_previsions,rdt,eta=0.25):
    """
   Trimmed mean de Timmermann(2006)
    """
    
    # Calcul du nombre de lignes
    t=M_previsions.shape[0]
    # Calcul du nombre de colonnes
    M=M_previsions.shape[1]
    
    # Création d'une matrice contenant les pertes de tick des différents modèles de VaR
    tick_losses=np.zeros(M)
    
    # Calcul de la perte par rapport aux réalisations du portefeuille
    for i in range(0,M):
        for j in range(0,t):
            tick_losses[i]=tick_losses[i]+abs(rdt[j]-M_previsions[j][i])
    
    # Détermination du classement de chaques modèle en fonction des pertes
    # Le modèle avec la perte la plus faible sera le mieux classé
    argsort_array=(tick_losses).argsort()
    
    ranks_array = np.empty_like(argsort_array)
    ranks_array[argsort_array] = np.arange(len(tick_losses))
    
    # Création de la matrice de poids des betas
    beta_weight=np.zeros(M)
    
    # Boucle attribuant des poids seulement aux quatre meilleurs modèles
    for i in ranks_array:
        if (ranks_array[i]+1)<=np.floor(eta*M):
            beta_weight[i]=1/(eta*M)
        else:
            beta_weight[i]=0            
    
    return beta_weight


def  Simple_Mean(M_previsions,rdt):
    """
   Simple Mean
    """
    # Calcul du nombre de lignes
    t=M_previsions.shape[0]
    # Calcul du nombre de colonnes
    M=M_previsions.shape[1]

     # Création de la matrice de poids des betas
    beta_weight=np.zeros(M)
    
    # Boucle attribuant des poids proportionnels à tous les modèles
    for i in range(0,M):
        beta_weight[i]=1/M
    
    return beta_weight


def  Single_Best(M_previsions,rdt):
    """
   Single Best 
    """
    # Calcul du nombre de lignes
    t=M_previsions.shape[0]
    # Calcul du nombre de colonnes
    M=M_previsions.shape[1]
    
    # Création d'une matrice contenant les pertes de tick des différents modèles de VaR
    tick_losses=np.zeros(M)
    
    # Détermination du classement de chaques modèle en fonction des pertes
    # Le modèle avec la perte la plus faible sera le mieux classé
    for i in range(0,M):
        for j in range(0,t):
            tick_losses[i]=tick_losses[i]+abs(rdt[j]-M_previsions[j][i])

    argsort_array=(tick_losses).argsort()
    
    ranks_array = np.empty_like(argsort_array)
    ranks_array[argsort_array] = np.arange(len(tick_losses))
    
     # Création de la matrice de poids des betas
    beta_weight=np.zeros(M)
    
    # Boucle attribuant un poins uniquement au meilleur modèle
    for i in ranks_array:
        if (ranks_array[i])>0:
            beta_weight[i]=0
        else:
            beta_weight[i]=1
    
    return beta_weight


def Inverse_Loss(M_previsions,rdt):
    """
   Inverse loss de Timmermann(2006)
    """
    
    # Calcul du nombre de lignes
    t=M_previsions.shape[0]
    # Calcul du nombre de colonnes
    M=M_previsions.shape[1]
    
    # Création d'une matrice contenant les pertes de tick des différents modèles de VaR
    tick_losses=np.zeros(M)
    
    # Calcul de la perte par rapport aux réalisations du portefeuille
    for i in range(0,M):
        for j in range(0,t):
            tick_losses[i]=tick_losses[i]+abs(rdt[j]-M_previsions[j][i])
    
    # Calcul de la perte inverse des modèles
    model_losses=np.reciprocal(tick_losses)
    
    # Calcul de la la somme totale de la perte inverse des modèles
    total_losses=np.sum(np.reciprocal(tick_losses))
    
    # Calcul du Beta des modèles
    beta_weight=np.divide(model_losses,total_losses)
    
    return beta_weight


def Inverse_Rank(M_previsions,rdt):
    """
   Inverse rank de Timmermann(2006)
    """
    
    # Calcul du nombre de lignes
    t=M_previsions.shape[0]
    # Calcul du nombre de colonnes
    M=M_previsions.shape[1]
    
    # Création d'une matrice contenant les pertes de tick des différents modèles de VaR
    tick_losses=np.zeros(M)
    
    # Calcul de la perte par rapport aux réalisations du portefeuille
    for i in range(0,M):
        for j in range(0,t):
            tick_losses[i]=tick_losses[i]+abs(rdt[j]-M_previsions[j][i])
    
    
    # Détermination du classement de chaques modèle en fonction des pertes
    # Le modèle avec la perte la plus faible sera le mieux classé
    argsort_array=(tick_losses).argsort()
    
    ranks_array = np.empty_like(argsort_array)
    ranks_array[argsort_array] = np.arange(len(tick_losses))
    
    ranking_inverse=np.zeros(M)
    
    for i in ranks_array:
        if (ranks_array[i]+1)>0:
            ranking_inverse[i]=1/(ranks_array[i]+1)
        else:
            ranking_inverse[i]=0
     

    # Calcul de la la somme totale du classement inverse des modèles
    sum_ranking_inverse=np.sum(ranking_inverse)
    
    # Calcul du Beta des modèles
    beta_weight=np.divide(ranking_inverse,sum_ranking_inverse)

    return beta_weight


