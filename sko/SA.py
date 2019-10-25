#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/17
# @Author  : github.com/guofei9987

import numpy as np
import types

class SA:
    """
    DO SA(Simulated Annealing)

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    x0 : array, shape is n_dim
        initial solution
    max_iter : int
        Max of iter
    T :float
        initial temperature
    T_min : float
        end temperature
    L : float
        num of iteration under every temperature（Long of Chain）
    q : float
        cool down speed

    Attributes
    ----------------------


    Examples
    -------------
    >>> demo_func=lambda x: x[0]**2 + x[1]**2 + x[2]**2
    >>> from sko.SA import SA
    >>> sa = SA(func=demo_func, x0=[1, 1, 1])
    >>> x_star, y_star = sa.fit()
    """

    def __init__(self, func, x0, T=100, T_min=1e-7, L=300, q=0.9):
        self.func = func
        self.x = np.array(x0)  # initial solution
        self.T = T  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = L  # num of iteration under every temperature（Long of Chain）
        self.q = q  # cool down speed
        self.f_list = []  # history
        self.x_star, self.f_star = None, None

    def new_x(self, x):
        return 0.2 * np.random.randn(len(x)) + x

    def fit(self):
        func = self.func
        T = self.T
        T_min = self.T_min
        L = self.L
        q = self.q
        x = self.x

        f1 = func(x)
        self.x_star, self.f_star = x, f1  # 全局最优
        while T > T_min:
            for i in range(L):
                # 随机扰动
                x2 = self.new_x(x)
                f2 = func(x2)

                # 加入到全局列表
                if f2 < self.f_star:
                    self.x_star, self.f_star = x2, f2
                self.f_list.append(f2)

                # Metropolis
                df = f2 - f1
                if df < 0 or np.exp(-df / T) > np.random.rand():
                    x, f1 = x2, f2

            T = T * q  # 降温
        return self.x_star, self.f_star

    run = fit

    def register(self, operator_name, operator, *args, **kwargs):
        '''
        regeister udf to the class
        :param operator_name: string in {'crossover', 'mutation', 'selection', 'ranking'}
        :param operator: a function
        :param args: arg of operator
        :param kwargs: kwargs of operator
        :return:
        '''
        valid_operator_name = {'new_x'}
        if operator_name not in valid_operator_name:
            raise NameError(operator_name + "is not a valid operator name, should be in " + str(valid_operator_name))

        def operator_wapper(*wrapper_args):
            return operator(*(wrapper_args + args), **kwargs)

        setattr(self, operator_name, types.MethodType(operator_wapper, self))
        return self


class SA_TSP(SA):
    def new_x(self, x):
        x = x.copy()
        n1, n2 = np.random.randint(0, len(x), 2)
        x[n1], x[n2] = x[n2], x[n1]
        return x


def sa_register_udf(udf_func_dict):
    class SAUdf(SA):
        pass

    for udf_name in udf_func_dict:
        if udf_name == 'new_x':
            SAUdf.new_x = udf_func_dict[udf_name]
    return SAUdf
