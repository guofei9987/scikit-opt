#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987


import numpy as np
from .base import SkoBase
from sko.tools import func_transformer
from abc import ABCMeta, abstractmethod


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


# %% operators:

def ranking_raw(self):
    # GA select the biggest one, but we want to minimize func, so we put a negative here
    self.FitV = -self.Y
    return self.FitV


def ranking_linear(self):
    '''
    For more details see [Baker1985]_.

    :param self:
    :return:

    .. [Baker1985] Baker J E, "Adaptive selection methods for genetic
    algorithms, 1985.
    '''
    self.FitV = np.argsort(np.argsort(-self.Y))
    return self.FitV


def selection_tournament(self, tourn_size=3):
    '''
    Select the best individual among *tournsize* randomly chosen
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    FitV = self.FitV
    sel_index = []
    for i in range(self.size_pop):
        # aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        aspirants_index = np.random.randint(self.size_pop, size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom


def selection_tournament_faster(self, tourn_size=3):
    '''
    Select the best individual among *tournsize* randomly chosen
    Same with `selection_tournament` but much faster using numpy
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    aspirants_idx = np.random.randint(self.size_pop, size=(self.size_pop, tourn_size))
    aspirants_values = self.FitV[aspirants_idx]
    winner = aspirants_values.argmax(axis=1)  # winner index in every team
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom


def selection_roulette_1(self):
    '''
    Select the next generation using roulette
    :param self:
    :return:
    '''
    FitV = self.FitV
    FitV = FitV - FitV.min() + 1e-10
    # the worst one should still has a chance to be selected
    sel_prob = FitV / FitV.sum()
    sel_index = np.random.choice(range(self.size_pop), size=self.size_pop, p=sel_prob)
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom


def selection_roulette_2(self):
    '''
    Select the next generation using roulette
    :param self:
    :return:
    '''
    FitV = self.FitV
    FitV = (FitV - FitV.min()) / (FitV.max() - FitV.min() + 1e-10) + 0.2
    # the worst one should still has a chance to be selected
    sel_prob = FitV / FitV.sum()
    sel_index = np.random.choice(range(self.size_pop), size=self.size_pop, p=sel_prob)
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom


def crossover_1point(self):
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    for i in range(0, size_pop, 2):
        n = np.random.randint(0, self.len_chrom, 1)
        # crossover at the point n
        seg1, seg2 = self.Chrom[i, n:].copy(), self.Chrom[i + 1, n:].copy()
        self.Chrom[i, n:], self.Chrom[i + 1, n:] = seg2, seg1
    return self.Chrom


def crossover_2point(self):
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    for i in range(0, size_pop, 2):
        n1, n2 = np.random.randint(0, self.len_chrom, 2)
        if n1 > n2:
            n1, n2 = n2, n1
        # crossover at the points n1 to n2
        seg1, seg2 = self.Chrom[i, n1:n2].copy(), self.Chrom[i + 1, n1:n2].copy()
        self.Chrom[i, n1:n2], self.Chrom[i + 1, n1:n2] = seg2, seg1
    return self.Chrom


def crossover_2point_bit(self):
    '''
    3 times faster than `crossover_2point`, but only use for 0/1 type of Chrom
    :param self:
    :return:
    '''
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    half_size_pop = int(size_pop / 2)
    Chrom1, Chrom2 = Chrom[:half_size_pop], Chrom[half_size_pop:]
    mask = np.zeros(shape=(half_size_pop, len_chrom), dtype=int)
    for i in range(half_size_pop):
        n1, n2 = np.random.randint(0, self.len_chrom, 2)
        if n1 > n2:
            n1, n2 = n2, n1
        mask[i, n1:n2] = 1
    mask2 = (Chrom1 ^ Chrom2) & mask
    Chrom1 ^= mask2
    Chrom2 ^= mask2
    return self.Chrom


# def crossover_rv_3(self):
#     Chrom, size_pop = self.Chrom, self.size_pop
#     i = np.random.randint(1, self.len_chrom)  # crossover at the point i
#     Chrom1 = np.concatenate([Chrom[::2, :i], Chrom[1::2, i:]], axis=1)
#     Chrom2 = np.concatenate([Chrom[1::2, :i], Chrom[0::2, i:]], axis=1)
#     self.Chrom = np.concatenate([Chrom1, Chrom2], axis=0)
#     return self.Chrom


def mutation(self):
    '''
    mutation of 0/1 type chromosome
    faster than `self.Chrom = (mask + self.Chrom) % 2`
    :param self:
    :return:
    '''
    #
    mask = (np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut)
    self.Chrom ^= mask
    return self.Chrom


def crossover_pmx(self):
    '''
    Executes a partially matched crossover (PMX) on Chrom.
    For more details see [Goldberg1985]_.

    :param self:
    :return:

    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
   salesman problem", 1985.
    '''
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    for i in range(0, size_pop, 2):
        Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
        cxpoint1, cxpoint2 = np.random.randint(0, self.len_chrom - 1, 2)
        if cxpoint1 >= cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1 + 1
        # crossover at the point cxpoint1 to cxpoint2
        pos1_recorder = {value: idx for idx, value in enumerate(Chrom1)}
        pos2_recorder = {value: idx for idx, value in enumerate(Chrom2)}
        for j in range(cxpoint1, cxpoint2):
            value1, value2 = Chrom1[j], Chrom2[j]
            pos1, pos2 = pos1_recorder[value1], pos2_recorder[value2]
            Chrom1[j], Chrom1[pos1] = Chrom1[pos1], Chrom1[j]
            Chrom2[j], Chrom2[pos2] = Chrom2[pos2], Chrom2[j]
            pos1_recorder[value1], pos1_recorder[value2] = pos1, j
            pos2_recorder[value1], pos2_recorder[value2] = j, pos2

        self.Chrom[i], self.Chrom[i + 1] = Chrom1, Chrom2
    return self.Chrom


def mutation_TSP_1(self):
    for i in range(self.size_pop):
        for j in range(self.n_dim):
            if np.random.rand() < self.prob_mut:
                n = np.random.randint(0, self.len_chrom, 1)
                self.Chrom[i, j], self.Chrom[i, n] = self.Chrom[i, n], self.Chrom[i, j]
    return self.Chrom


def mutation_TSP_3(self):
    for i in range(self.size_pop):
        if np.random.rand() < self.prob_mut:
            n1, n2 = np.random.randint(0, self.len_chrom, 2)
            self.Chrom[i, n1], self.Chrom[i, n2] = self.Chrom[i, n2], self.Chrom[i, n1]
    return self.Chrom


# %%

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
                 prob_mut=0.001, **kwargs):
        self.func = func_transformer(func)
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim

        self.define_chrom(kwargs)
        self.crtbp(self.size_pop, self.len_chrom)

        self.X = None  # shape is size_pop, n_dim (it is all x of func)
        self.Y = None  # shape is size_pop,
        self.FitV = None

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []
        self.all_history_Y = []
        # self.generation_best_ranking = []

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

    ranking = ranking_raw
    selection = selection_tournament_faster
    crossover = crossover_2point_bit
    mutation = mutation

    def run(self):
        for i in range(self.max_iter):
            self.X = self.chrom2x()
            self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()

            # record the best ones
            generation_best_index = self.Y.argmin()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)

        general_best_index = (np.array(self.generation_best_Y)).argmin()
        general_best_X, general_best_Y = \
            self.generation_best_X[general_best_index], self.generation_best_Y[general_best_index]
        return general_best_X, general_best_Y

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

    crossover = crossover_pmx
    mutation = mutation_TSP_1
