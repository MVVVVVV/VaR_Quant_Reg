import numpy as np
from scipy.optimize import minimize


# https://github.com/Buczman/CaviaR
# modifié et traduit du R


def SAV(beta, rdt, var_start, var_predict):
    """
    Symmetric absolute value
    """
    var = np.array([var_start])
    size = rdt.size

    for i in range(1, size):
        var[i] = beta[0] + beta[1] * var[i - 1] + beta[2] * abs(rdt[i - 1])

    if var_predict:
        var[size] = beta[0] + beta[1] * var[size] + beta[2] * abs(rdt[size - 1])

    return var


def AS(beta, rdt, var_start, var_predict):
    """
    Asymmetric slope
    """
    var = np.array([var_start])
    size = rdt.size

    for i in range(1, size):
        var[i] = beta[0] + beta[1] * var[i - 1] + beta[2] * max(rdt[i - 1], 0) + beta[3] * min(rdt[i - 1], 0)

    if var_predict:
        var[size] = beta[0] + beta[1] * var[size - 1] + beta[2] * max(rdt[size - 1], 0) + beta[3] * min(rdt[size - 1],
                                                                                                        0)

    return var


def IG(beta, rdt, var_start, var_predict):
    """
    Indirect GARCH(1; 1)
    """
    var = np.array([var_start])
    size = rdt.size

    for i in range(1, size):
        var[i] = (beta[0] + beta[1] * var[i - 1] ** 2 + beta[2] * rdt[i - 1] ** 2) ** 0.5

    if var_predict:
        var[size] = (beta[0] + beta[1] * var[size - 1] ** 2 + beta[2] * rdt[size - 1] ** 2) ** 0.5

    return var


def AD(beta, rdt, var_start, var_predict, var_quantile=0.05, G=10):
    """
    Adaptive
    G is by default set to 10 as in Engle and Manganelli (2004)
    """
    var = np.array([var_start])
    size = rdt.size

    for i in range(1, size):
        var[i] = var[i - 1] + beta[0] * (1 / (1 + np.exp(G * (rdt[i - 1] - var[i - 1]))) - var_quantile)
        # var_quantile etre sur??

    if var_predict:
        var[size] = var[size - 1] + beta[0] * (1 / (1 + np.exp(G * (rdt[size - 1] - var[size - 1]))) - var_quantile)
    return var


def CaviarOptim(rdt, var_quantile=0.05, model=1):
    emp_var = np.quantile(rdt[1:300], var_quantile)  # le papier recommande de prendre les 300 premieres
    size = np.size(rdt)
    var = np.zeros(size)
    hit = np.zeros(size)

    # on initialise nos betas
    if model == 1 or model == 3:
        n_beta = 3
        n_initialcond = 10
    elif model == 2:
        n_beta = 4
        n_initialcond = 15
    elif model == 4:
        n_beta = 1
        n_initialcond = 5

    # on crée plusieurs conditions initiales, on selectionnera le meilleur resultat plus tard
    params_start = np.random.random((n_initialcond, n_beta))
    res_opt = np.zeros(n_initialcond, n_beta + 1)

    for i in range(1, n_initialcond + 1):
        args = [1, model, rdt, var_quantile, emp_var]
        res = minimize(fun=ObjectiveFunction, x0=params_start[i, :], args=args)
        res_opt[i, 1] = res.fun
        res_opt[i, 2:(n_beta + 1)] = np.array(res.x)


def ObjectiveFunction(beta, out, model, rdt, var_quantile, emp_quantile, var_predict=False):
    if var_predict:
        var = np.zeros(0, np.size(rdt) + 1)
    else:
        var = np.zeros(0, np.size(rdt))
        hit = var

    var[0] = - emp_quantile  # pourquoi le -??? a retirer prob

    if model == "SAV":
        var = SAV(beta, rdt, var[0], var_predict)
    elif model == "AS":
        var = AS(beta, rdt, var[0], var_predict)
    elif model == "IG":
        var = IG(beta, rdt, var[0], var_predict)
    elif model == "AD":
        var = AD(beta, rdt, var[0], var_predict)

    if not var_predict:
        hit = (rdt < -var) - var_quantile
        if out == 1:
            RQ = -1 * np.transpose(hit) * (rdt + var)
            if RQ > 10 ** 10:
                RQ = 10 ** 10

            return RQ

        elif out == 2:
            return var, hit

    elif var_predict:
        return var
