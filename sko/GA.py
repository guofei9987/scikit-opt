#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : @guofei9987


import numpy as np


class GA:
    """
    Do genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
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
    >>> best_x, best_y = ga.fit()
    """

    # genetic algorithms
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001, **kwargs):
        self.func = func
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        self.define_chrom(kwargs)
        self.crtbp(self.size_pop, self.len_chrom)

        self.X = self.chrom2x()  # shape is size_pop, n_dim (it is all x of func)
        self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_ranking = []

    def define_chrom(self, kwargs):
        # define the types of Chrom
        self.lb = kwargs.get('lb', [-1] * self.n_dim)
        self.ub = kwargs.get('ub', [-1] * self.n_dim)
        self.precision = kwargs.get('precision', [1e-7] * self.n_dim)

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

    def ranking(self, X):
        # GA select the biggest one, but we want to minimize func, so we put a negative here
        FitV = np.array([-self.func(x) for x in X])
        return FitV

    def selection(self, FitV):
        # do Roulette to select the next generation
        # FitV = FitV - FitV.min() + 1e-10
        FitV = (FitV - FitV.min()) / (FitV.max() - FitV.min() + 1e-10) + 0.2
        # the worst one should still has a chance to be selected
        sel_prob = FitV / FitV.sum()
        sel_index = np.random.choice(range(self.size_pop), size=self.size_pop, p=sel_prob)
        self.Chrom = self.Chrom[sel_index, :]  # next generation
        return self.Chrom

    def crossover(self):
        Chrom, size_pop = self.Chrom, self.size_pop
        i = np.random.randint(1, self.len_chrom)  # crossover at the point i
        Chrom1 = np.concatenate([Chrom[::2, :i], Chrom[1::2, i:]], axis=1)
        Chrom2 = np.concatenate([Chrom[1::2, :i], Chrom[0::2, i:]], axis=1)
        self.Chrom = np.concatenate([Chrom1, Chrom2], axis=0)
        return self.Chrom

    def mutation(self):
        # mutation
        mask = (np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut) * 1
        self.Chrom = (mask + self.Chrom) % 2
        return self.Chrom

    def fit(self):
        max_iter = self.max_iter
        func = self.func
        for i in range(max_iter):
            X = self.chrom2x()
            FitV = self.ranking(X)
            self.selection(FitV)
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_X = X[FitV.argmax(), :]
            self.generation_best_X.append(generation_best_X)
            self.generation_best_ranking.append(FitV.max())
            self.FitV_history.append(FitV)
        general_best = self.generation_best_X[(np.array(self.generation_best_ranking)).argmax()]
        return general_best, func(general_best)


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

    def crossover(self):
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        for i in range(0, int(size_pop / 2), 2):
            Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
            n1, n2 = np.random.randint(0, self.len_chrom, 2)
            n1, n2 = min(n1, n2), max(n1, n2)
            # crossover at the point n1 to n2
            for j in range(n1, n2):
                x = np.argwhere(Chrom1 == Chrom2[j])
                y = np.argwhere(Chrom2 == Chrom1[j])
                Chrom1[j], Chrom2[j] = Chrom2[j], Chrom1[j]
                Chrom1[x], Chrom2[y] = Chrom2[y], Chrom1[x]
            self.Chrom[i], self.Chrom[i + 1] = Chrom1, Chrom2
        return self.Chrom

    def mutation(self):
        for i in range(self.size_pop):
            if np.random.rand() < self.prob_mut:
                n1, n2 = np.random.randint(0, self.len_chrom, 2)
                self.Chrom[i, n1], self.Chrom[i, n2] = self.Chrom[i, n2], self.Chrom[i, n1]
        return self.Chrom


def register_udf(udf_func_dict):
    class GAUdf(GA):
        pass

    for udf_name in udf_func_dict:
        if udf_name == 'crossover':
            GAUdf.crossover = udf_func_dict[udf_name]
        elif udf_name == 'mutation':
            GAUdf.mutation = udf_func_dict[udf_name]
        elif udf_name == 'selection':
            GAUdf.selection = udf_func_dict[udf_name]
        elif udf_name == 'ranking':
            GAUdf.ranking = udf_func_dict[udf_name]
    return GAUdf
