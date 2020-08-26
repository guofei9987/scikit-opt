import numpy as np


def sphere(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put


def schaffer(p):
    '''
    这个函数是二维的复杂函数，具有无数个极小值点
    在(0,0)处取的最值0
    这个函数具有强烈的震荡形态，所以很难找到全局最优质值
    :param p:
    :return:
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


def myfunc(p):
    x = p
    return np.sin(x) + np.square(x)
