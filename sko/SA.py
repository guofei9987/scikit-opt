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
    See https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py
    """

    def __init__(self, func, x0, T_max=100, T_min=1e-7, L=300, q=0.9, max_stay_counter=150):
        assert T_max > T_min > 0, 'T_max > T_min > 0'
        assert 0 < q < 1, '0<q<1'
        self.func = func

        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（Long of Chain）
        self.q = q  # cool down speed
        self.max_stay_counter = max_stay_counter  # stop if best_y stay unchanged over max_stay_counter times

        self.best_x = np.array(x0)  # initial solution
        self.best_y = self.func(self.best_x)
        self.T = self.T_max
        self.iter_cycle = 0
        self.best_y_history = [self.best_y]
        self.best_x_history = [self.best_x]

    def get_new_x(self, x):
        if np.random.rand()>0.1:
            return 0.2 * self.T * np.random.randn(len(x)) + x
        else:
            return  0.2 * self.T * np.random.randn(len(x)) + self.best_x

    def cool_down(self):
        self.T *= self.q

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0
        while True:
            for i in range(self.L):
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                # Metropolis
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.best_y_history.append(self.best_y)
            self.best_x_history.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                stop_code = 'Cooled to final temperature'
                break
            if stay_counter > self.max_stay_counter:
                stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break

        return self.best_x, self.best_y

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
            # randomly generate n1 < n2 < n3. Notice: not equal
            n1, n2, n3 = sorted(np.random.randint(0, len(x_new) - 2, 3))
            n2 += 1
            n3 += 2
            slice1, slice2, slice3, slice4 = x_new[0:n1], x_new[n1:n2], x_new[n2:n3 + 1], x_new[n3 + 1:]
            x_new = np.concatenate([slice1, slice3, slice2, slice4])
            return x_new

        new_x_strategy = np.random.randint(3)
        if new_x_strategy == SWAP:
            x_new = swap(x_new)
        elif new_x_strategy == REVERSE:
            x_new = reverse(x_new)
        elif new_x_strategy == TRANSPOSE:
            x_new = transpose(x_new)

        return x_new
