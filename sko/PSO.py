#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
import matplotlib.pyplot as plt
from sko.tools import func_transformer


class PSO:
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations

    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    >>> demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
    >>> pso = PSO(func=demo_func, dim=3)
    >>> gbest_x, gbest_y = pso.fit()
    >>> print('best_x is ', pso.gbest_x, 'best_y is ', pso.gbest_y)
    >>> pso.plot_history()
    """

    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None):
        self.func = func_transformer(func)
        self.w = 0.8  # inertia
        self.cp, self.cg = 0.5, 0.5  # parameters to control personal best,global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        if lb is None and ub is None:
            self.is_bounded = False
        else:
            self.is_bounded = True
            self.lb = lb or [-np.inf] * dim  # lower bound
            self.ub = ub or [np.inf] * dim  # upper bound

        self.X = np.random.rand(self.pop, self.dim)  # location of particles, which is the value of variables of func
        self.V = np.random.rand(self.pop, self.dim)  # speed of particles
        self.Y = self.cal_y()  # image of function corresponding to every particles for one generation
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles in history
        self.gbest_y = np.inf  # general best image  for all particles in history
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

    def cal_y(self):
        # calculate y for every x in X
        self.Y = np.array([self.func(x) for x in self.X]).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        Best for individual
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        Best for the population
        :return:
        '''
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :]
            self.gbest_y = self.Y.min()

    def run(self):
        for iter_num in range(self.max_iter):
            r1 = np.random.rand(self.pop, self.dim)
            r2 = np.random.rand(self.pop, self.dim)
            self.V = self.w * self.V + \
                     self.cp * r1 * (self.pbest_x - self.X) + \
                     self.cg * r2 * (self.gbest_x - self.X)
            self.X = self.X + self.V

            if self.is_bounded:  # with constraints
                self.X = np.clip(self.X, self.lb, self.ub)

            self.cal_y()

            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    def plot_history(self):
        plt.plot(self.gbest_y_hist)
        plt.show()

    fit = run
