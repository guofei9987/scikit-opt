#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : @guofei9987

import numpy as np
import matplotlib.pyplot as plt


class PSO:
    """
    Do PSO(Particle swarm optimization)

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep pace with GA
    max_iter : int
        Max of iter

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
    >>>demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
    >>>pso = PSO(func=demo_func, dim=3)
    >>>pso.fit()
    >>>print('best_x is ', pso.gbest_x)
    >>>print('best_y is ', pso.gbest_y)
    >>>pso.plot_history()
    """
    def __init__(self, func, dim, pop=40, max_iter=150):
        self.func = func
        self.w = 0.8  # 惯性权重
        self.cp, self.cg = 2, 2  # 加速常数，一般取2附近. p代表个体记忆，g代表集体记忆
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.X = np.random.rand(self.pop, self.dim)  # location of particles, which is the value of variables of func
        self.V = np.random.rand(self.pop, self.dim)  # speed of particles
        self.Y = self.cal_y()  # image of function corresponding to every particles for one generation
        self.pbest_x = self.X.copy()  # best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # general best location for all particles in history
        self.gbest_y = np.inf  # general best image  for all particles in history
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

    def cal_y(self):
        # calculate y for every x in X
        self.Y = np.array([self.func(x) for x in self.X]).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :]
            self.gbest_y = self.Y.min()

    def fit(self):
        for iter_num in range(self.max_iter):
            # self.dim
            r1 = np.random.rand(self.pop, self.dim)
            r2 = np.random.rand(self.pop, self.dim)
            self.V = self.w * self.V + \
                     self.cp * r1 * (self.pbest_x - self.X) + \
                     self.cg * r2 * (self.gbest_x - self.X)
            self.X = self.X + self.V
            self.cal_y()

            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    def plot_history(self):
        plt.plot(self.gbest_y_hist)
        plt.show()
