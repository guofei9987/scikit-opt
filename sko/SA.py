#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/17
# @Author  : github.com/guofei9987

import numpy as np
import types
from .base import SkoBase


class SA(SkoBase):
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
    T_max :float
        initial temperature
    T_min : float
        end temperature
    L : int
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

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, q=0.9):
        assert T_max > 0, 'T_max>0'
        assert T_min > 0, 'T_min>0'
        assert 0 < q < 1, '0<q<1'
        self.func = func

        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（Long of Chain）
        self.q = q  # cool down speed

        self.x_best = np.array(x0)  # initial solution
        self.y_best = self.func(self.x_best)
        self.T = self.T_max
        self.iter_cycle = 0
        self.y_best_history = [self.y_best]

    def get_new_x(self, x):
        return 0.2 * np.random.randn(len(x)) + x

    def cool_down(self):
        self.T *= self.q

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-09):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current = self.x_best, self.y_best
        max_stay_counter = 150
        stay_counter = 0
        while self.T > self.T_min and stay_counter <= max_stay_counter:
            for i in range(self.L):
                # 随机扰动
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                # Metropolis
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.y_best:
                        self.x_best, self.y_best = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.y_best_history.append(self.y_best)

            # 连续多少次没有变优，就停止迭代
            if self.isclose(self.y_best_history[-1], self.y_best_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

        return self.x_best, self.y_best

    fit = run


class SA_TSP(SA):
    def cool_down(self):
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))

    def get_new_x(self, x):
        x_new = x.copy()
        SWAP, REVERSE, TRANSPOSE = 0, 1, 2

        def swap(x_new):
            n1, n2 = np.random.randint(0, len(x_new) - 1, 2)
            if n1 >= n2:
                n1, n2 = n2, n1 + 1
            x_new[n1], x_new[n2] = x_new[n2], x_new[n1]
            return x_new

        def reverse(x_new):
            n1, n2 = np.random.randint(0, len(x_new) - 1, 2)
            if n1 >= n2:
                n1, n2 = n2, n1 + 1
            x_new[n1:n2] = x_new[n1:n2][::-1]

            return x_new

        def transpose(x_new):
            while True:
                n1, n2, n3 = np.random.randint(0, len(x_new), 3)

                if n1 != n2 != n3 != n1:
                    break
            # Let n1 < n2 < n3
            n1, n2, n3 = sorted([n1, n2, n3])

            # Insert data between [n1,n2) after n3
            tmplist = x_new[n1:n2].copy()
            x_new[n1: n1 + n3 - n2 + 1] = x_new[n2: n3 + 1].copy()
            x_new[n3 - n2 + 1 + n1: n3 + 1] = tmplist.copy()
            return x_new

        new_x_strategy = np.random.randint(3)
        if new_x_strategy == SWAP:
            x_new = swap(x_new)
        elif new_x_strategy == REVERSE:
            x_new = reverse(x_new)
        elif new_x_strategy == TRANSPOSE:
            x_new = transpose(x_new)

        return x_new
