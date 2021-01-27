import numpy as np
from scipy.optimize import minimize
import pandas as pd
import itertools
import time
from numba import jit
import math


@jit(nopython=True)
def SAV(beta, rdt, var, var_predict):
    """
    Symmetric absolute value
    """
    size = rdt.size

    for i in range(1, size):
        var[i] = beta[0] + beta[1] * var[i - 1] + beta[2] * abs(rdt[i - 1])

    if var_predict:
        var[size] = beta[0] + beta[1] * var[size - 1] + beta[2] * abs(rdt[size - 1])

    return var


@jit(nopython=True)
def AS(beta, rdt, var, var_predict):
    """
    Asymmetric slope
    """
    size = rdt.size

    for i in range(1, size):
        var[i] = beta[0] + beta[1] * var[i - 1] + beta[2] * max(rdt[i - 1], 0) + beta[3] * min(rdt[i - 1], 0)

    if var_predict:
        var[size] = beta[0] + beta[1] * var[size - 1] + beta[2] * max(rdt[size - 1], 0) + beta[3] * min(rdt[size - 1],
                                                                                                        0)

    return var


@jit(nopython=True)
def IG(beta, rdt, var, var_predict):
    """
    Indirect GARCH(1; 1)
    """
    size = rdt.size

    for i in range(1, size):
        var[i] = (beta[0] + beta[1] * var[i - 1] ** 2 + beta[2] * rdt[i - 1] ** 2) ** 0.5

    if var_predict:
        var[size] = (beta[0] + beta[1] * var[size - 1] ** 2 + beta[2] * rdt[size - 1] ** 2) ** 0.5

    return var


@jit(nopython=True)
def AD(beta, rdt, var, var_predict, var_quantile=0.01, G=10):
    """
    Adaptive
    G is by default set to 10 as in Engle and Manganelli (2004)
    """
    size = rdt.size

    for i in range(1, size):
        var[i] = var[i - 1] + beta[0] * ((1 + np.exp(G * (rdt[i - 1] - var[i - 1]))) ** -1 - var_quantile)
        # var_quantile etre sur??

    if var_predict:
        var[size] = var[size - 1] + beta[0] * ((1 + np.exp(G * (rdt[size - 1] - var[size - 1]))) ** -1 - var_quantile)
    return var


def CaviarOptim(rdt, var_quantile=0.01, model=1):
    emp_var = np.quantile(rdt[0:300], var_quantile)  # le papier recommande de prendre les 300 premieres
    size = np.size(rdt)
    var = np.zeros(size)
    hit = np.zeros(size)

    # on initialise nos betas
    if model == 1 or model == 3:
        n_beta = 3
    elif model == 2:
        n_beta = 4
    elif model == 4:
        n_beta = 1

    n = 0
    while n < 4:
        n += 1
        # on crée plusieurs conditions initiales, on selectionnera le meilleur resultat plus tard
        params_start = np.random.random(n_beta)
        res_opt = np.zeros(n_beta + 1)

        args = (1, model, rdt, var_quantile, emp_var)
        bnds = list(itertools.repeat((-2.00001, 2.000001), n_beta))
        res = minimize(fun=ObjectiveFunction, x0=params_start, args=args, method='Nelder-Mead')
        if model == 3:
            res = minimize(fun=ObjectiveFunction, x0=res.x, args=args, bounds=bnds)
        else:
            res = minimize(fun=ObjectiveFunction, x0=res.x, args=args, method='BFGS')

        fun_value = res.fun
        best_betas = res.x

        var = ObjectiveFunction(best_betas, 2, model, rdt, var_quantile, emp_var, True)
        var_predict = var[-1]
        var = var[:-1]

        if not math.isnan(var_predict):
            break

    return fun_value, best_betas, var, -var_predict


def ObjectiveFunction(beta, out, model, rdt, var_quantile, emp_quantile, var_predict=False):
    if var_predict:
        var = np.zeros((np.size(rdt) + 1))
    else:
        var = np.zeros(np.size(rdt))
        hit = var

    var[0] = -emp_quantile

    if model == 1:
        var = SAV(beta, rdt, var, var_predict)
    elif model == 2:
        var = AS(beta, rdt, var, var_predict)
    elif model == 3:
        var = IG(beta, rdt, var, var_predict)
    elif model == 4:
        var = AD(beta, rdt, var, var_predict)

    if not var_predict:
        hit = (rdt < -var) - var_quantile
        if out == 1:
            RQ = -1 * np.dot(np.transpose(hit), (rdt + var))
            if RQ == float("inf"):
                RQ = 100000000000

            return RQ

        elif out == 2:
            return var, hit

    elif var_predict:
        return var


if __name__ == '__main__':
    rdt = pd.read_csv(r'C:\Users\MV\Desktop\Projet Quant\rdt.csv', index_col=0)
    start_date = '2000-01-03'
    end_date = '2006-12-29'
    start_index = list(rdt.index).index(start_date)
    end_index = list(rdt.index).index(end_date)
    var_prediction = pd.DataFrame(columns=rdt.columns, index=rdt.index[start_index:end_index + 1])

    for model in [1,2,3,4]:
        for column in rdt:
            print(column)
            i = 0
            t0 = time.time()
            for date in rdt.index[start_index - 1:end_index]:
                my_rdt = rdt.loc[:date][column].tail(1000)
                my_rdt = np.squeeze(np.asarray(my_rdt))  # utilisable par numba
                fun_value, best_betas, var, var_prediction[column][i] = CaviarOptim(my_rdt, model=model)
                i += 1
            t1 = time.time()
            print(t1 - t0)
        var_prediction.fillna(method='ffill')
        var_prediction.to_csv(r'C:\Users\MV\Desktop\Projet Quant\caviar_' + str(model) + '_NM_BFGS_V3_2000.csv')
