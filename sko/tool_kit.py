import numpy as np
from sko.GA import GA


def x2gray(x, n_dim, lb, ub, precision):
    ga = GA(func=lambda k: None, n_dim=n_dim, size_pop=2, max_iter=1, lb=lb, ub=ub, precision=precision)

    ub = ga.ub_extend if ga.int_mode else ga.ub  # for int mode
    x = (x - ga.lb) / (ub - ga.lb)  # map to (0,1)
    x = np.round(x * (np.exp2(ga.Lind) - 1)).astype(int)  # map to int

    res = np.zeros((x.shape[0], ga.Lind.sum()))
    for row_idx, row in enumerate(x):
        tmp1 = ''
        for col_idx, col in enumerate(row):
            tmp2 = bin(col ^ (col >> 1))[2:]  # real value to gray code
            tmp2 = '0' * (ga.Lind[col_idx] - len(tmp2)) + tmp2  # zero fill
            tmp1 += tmp2
        res[row_idx, :] = (np.array(list(tmp1)) == '1') * 1

    return res
