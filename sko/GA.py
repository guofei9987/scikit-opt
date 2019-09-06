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
    pop : int
        Size of population
    max_iter : int
        Max of iter
    Pm : float between 0 and 1
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
    >>>demo_func=lambda x: x[0]**2 + x[1]**2 + x[2]**2
    >>>ga = GA(func=demo_func, lb=[-1, -10, -5], ub=[2, 10, 2], max_iter=500)
    >>>best_x, best_y = ga.fit()
    """
    # genetic algorithms
    def __init__(self, func,
                 lb=[-1, -10, -5], ub=[2, 10, 2],
                 precision=None, pop=50, max_iter=200,
                 Pm=0.001):
        self.func = func
        self.pop = pop  # size of population
        self.max_iter = max_iter
        self.lb = lb  # a list of lower bound of each variable
        self.ub = ub  # a list of upper bound of each variable
        self.Pm = Pm  # probability of mutation
        precision = precision or [None for i in lb]
        precision = np.array([i or 1e-7 for i in precision])
        self.precision = precision
        # Lind is the num of genes of every variable of func（segments）
        Lind = np.ceil(np.log2((np.array(ub) - np.array(lb)) / np.array(precision))) + 1
        self.Lind = np.array([int(i) for i in Lind])
        self.total_Lind = int(sum(self.Lind))
        self.crtbp(self.pop, self.total_Lind)
        self.X = None  # every row is a value of variable of func with respect to one individual in the population
        self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_ranking = []

    def crtbp(self, pop=10, total_Lind=30):
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(pop, total_Lind))
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
        # do Roulette to select the best ones
        # FitV = FitV - FitV.min() + 1e-10
        FitV = (FitV - FitV.min())/(FitV.max() - FitV.min()+1e-10) + 0.2
        # the worst one should still has a chance to be selected
        sel_prob = FitV / FitV.sum()
        sel_index = np.random.choice(range(self.pop), size=self.pop, p=sel_prob)
        self.Chrom = self.Chrom[sel_index, :]  # next generation
        return self.Chrom

    def crossover(self):
        Chrom, pop = self.Chrom, self.pop
        i = np.random.randint(1, self.total_Lind)  # crossover at the point i
        Chrom1 = np.concatenate([Chrom[::2, :i], Chrom[1::2, i:]], axis=1)
        Chrom2 = np.concatenate([Chrom[1::2, :i], Chrom[0::2, i:]], axis=1)
        self.Chrom = np.concatenate([Chrom1, Chrom2], axis=0)
        return self.Chrom

    def mutation(self):
        # mutation
        mask = (np.random.rand(self.pop, self.total_Lind) < self.Pm) * 1
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
    pop : int
        Size of population
    max_iter : int
        Max of iter
    Pm : float between 0 and 1
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
    Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
    ```py
    import numpy as np

    num_points = 8

    points = range(num_points)
    points_coordinate = np.random.rand(num_points, 2)
    distance_matrix = np.zeros(shape=(num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i][j] = np.linalg.norm(points_coordinate[i] - points_coordinate[j], ord=2)
    print('distance_matrix is: \n', distance_matrix)


    def cal_total_distance(points):
        num_points, = points.shape
        total_distance = 0
        for i in range(num_points - 1):
            total_distance += distance_matrix[points[i], points[i + 1]]
        total_distance += distance_matrix[points[i + 1], points[0]]
        return total_distance
    ```

    Do GA
    ```py
    from GA import GA_TSP
    ga_tsp = GA_TSP(func=cal_total_distance, points=points, pop=50, max_iter=200, Pm=0.001)
    best_points, best_distance = ga_tsp.fit()
    ```
    """
    # genetic algorithms for TSP
    def __init__(self, func, points,
                 pop=50, max_iter=200,
                 Pm=0.001):
        self.func = func
        self.pop = pop  # size of population
        self.max_iter = max_iter
        self.Pm = Pm  # probability of mutation
        self.total_Lind = len(points)
        self.crtbp(self.pop, self.total_Lind)
        self.X = None  # every row is a value of variable of func with respect to one individual in the population
        self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_ranking = []

    def chrom2x(self):
        self.X = self.Chrom
        return self.X

    def crtbp(self, pop=10, total_Lind=30):
        # create the population
        tmp = np.random.rand(pop, total_Lind)
        self.Chrom = tmp.argsort(axis=1)
        return self.Chrom

    def crossover(self):
        Chrom, pop, total_Lind = self.Chrom, self.pop, self.total_Lind
        for i in range(0, int(pop / 2), 2):
            Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
            n1, n2 = np.random.randint(0, self.total_Lind, 2)
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
        # mutation
        for i in range(self.pop):
            if np.random.rand() < self.Pm:
                n1, n2 = np.random.randint(0, self.total_Lind, 2)
                n1, n2 = min(n1, n2), max(n1, n2)
                self.Chrom[i, n1], self.Chrom[i, n2] = self.Chrom[i, n2], self.Chrom[i, n1]
        return self.Chrom

