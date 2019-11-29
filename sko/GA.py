#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987


import numpy as np
from .base import SkoBase
from sko.tools import func_transformer
from abc import ABCMeta, abstractmethod
from .operators import crossover, mutation, ranking, selection


class GA_base(SkoBase, metaclass=ABCMeta):
    @abstractmethod
    def ranking(self):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass



class GA(GA_base):
    """genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every vaiiables of func
    precision : array_like
        The precision of every vaiiables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    >>> demo_func=lambda x: x[0]**2 + x[1]**2 + x[2]**2
    >>> ga = GA(func=demo_func,n_dim=3, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])
    >>> best_x, best_y = ga.run()
    """

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 constraint_eq=None, constraint_ueq=None,
                 **kwargs):
        self.func = func_transformer(func)
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        self.has_constraint = constraint_eq is not None or constraint_ueq is not None
        self.constraint_eq = constraint_eq  # a list of unequal constraint functions with c[i] <= 0
        self.constraint_ueq = constraint_ueq  # a list of equal functions with ceq[i] = 0

        self.define_chrom(kwargs)
        self.crtbp(self.size_pop, self.len_chrom)

        self.X = None  # shape is size_pop, n_dim (it is all x of func)
        self.Y = None  # shape is size_pop,
        self.FitV = None

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []
        self.generation_best_FitV = []

        self.all_history_Y = []
        self.all_history_FitV = []

    def define_chrom(self, kwargs):
        # define the types of Chrom
        self.lb = kwargs.get('lb', [-1] * self.n_dim)
        self.ub = kwargs.get('ub', [1] * self.n_dim)
        self.precision = kwargs.get('precision', 1e-7)
        self.precision = self.precision if isinstance(self.precision, list) else [self.precision] * self.n_dim

        # Lind is the num of genes of every variable of func（segments）
        Lind = np.ceil(np.log2((np.array(self.ub) - np.array(self.lb)) / np.array(self.precision))) + 1
        self.Lind = np.array([int(i) for i in Lind])
        self.len_chrom = int(sum(self.Lind))

    def crtbp(self, size_pop=10, len_chrom=30):
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(size_pop, len_chrom))
        return self.Chrom

    def gray2rv(self, gray_code):
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        # gray_code = crtbp(4, 2),gray2rv(gray_code)
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self):
        # Chrom to the variables of func
        Chrom = self.Chrom
        Lind = self.Lind
        lb = self.lb
        ub = self.ub
        cumsum_len_segment = Lind.cumsum()
        X = np.zeros(shape=(Chrom.shape[0], len(Lind)))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            temp1 = self.gray2rv(Chrom_temp)
            X[:, i] = lb[i] + (ub[i] - lb[i]) * temp1
        self.X = X
        return self.X

    def x2y(self):
        '''
        calculate Y for every X
        :return:
        '''
        self.Y = np.array([self.func(x) for x in self.X])
        return self.Y

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

    def run(self):
        for i in range(self.max_iter):
            self.X = self.chrom2x()
            self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.generation_best_FitV.append(self.FitV[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

        global_best_index = np.array(self.generation_best_FitV).argmax()
        global_best_X, global_best_Y = \
            self.generation_best_X[global_best_index], self.generation_best_Y[global_best_index]
        return global_best_X, global_best_Y

    fit = run


class GA_TSP(GA):
    """
    Do genetic algorithm to solve the TSP (Travelling Salesman Problem)
    Parameters
    ----------------
    func : function
        The func you want to do optimal.
        It inputs a candidate solution(a routine), and return the costs of the routine.
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    Attributes
    ----------------------
    Lind : array_like
         The num of genes corresponding to every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
    ```py
    num_points = 8
    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    print('distance_matrix is: \n', distance_matrix)
    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
    ```
    Do GA
    ```py
    from sko.GA import GA_TSP
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=8, pop=50, max_iter=200, Pm=0.001)
    best_points, best_distance = ga_tsp.fit()
    ```
    """

    def define_chrom(self, kwargs):
        self.len_chrom = self.n_dim

    def chrom2x(self):
        self.X = self.Chrom
        return self.X

    def crtbp(self, size_pop=10, len_chrom=30):
        # create the population
        tmp = np.random.rand(size_pop, len_chrom)
        self.Chrom = tmp.argsort(axis=1)
        return self.Chrom

    crossover = crossover.crossover_pmx
    mutation = mutation.mutation_TSP_1
