import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GA:
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
        self.num_variables_func = len(lb)  # the num of variables of func
        precision = precision or [None for i in lb]
        precision = np.array([i or 1e-7 for i in precision])
        self.precision = precision
        # Lind is the num of genes of every variable of func（segments）
        Lind = np.ceil(np.log2((np.array(ub) - np.array(lb)) / np.array(precision))) + 1
        self.Lind = np.array([int(i) for i in Lind])
        self.total_Lind = int(sum(self.Lind))
        self.crtbp(self.pop, self.total_Lind)
        self.X = None  # every row is a value of variable of func with respect to one individual of the population
        self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_ranking = []

    def crtbp(self, pop=10, total_Lind=30):
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(pop, total_Lind))
        return self.Chrom

    def gray2rv(self, gray_code):
        # Gray Code to real value
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        # gray_code = crtbp(4, 2),gray2rv(gray_code)
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def bs2rv(self):
        # Gray Code to real value
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

    def ranking(self):
        # GA select the biggest one, but we want to minimize func, so we put a negative here
        func, X = self.func, self.X
        FitV = np.array([-func(x) for x in X])
        return FitV

    def selection(self, FitV):
        # do Roulette to select the best ones
        # FitV: 适应度
        # index
        # sel_index=[]
        pop, = FitV.shape
        FitV = FitV - FitV.min() + 1e-10
        sel_prob = FitV / FitV.sum()
        sel_index = np.random.choice(range(pop), size=pop, p=sel_prob)
        return sel_index

    def crossover(self):
        # 奇数个的处理
        Chrom, pop = self.Chrom, self.pop
        i = int(pop / 2)  # crossover in the point i
        Chrom1 = np.concatenate([Chrom[::2, :i], Chrom[1::2, i:]], axis=1)
        Chrom2 = np.concatenate([Chrom[1::2, :i], Chrom[0::2, i:]], axis=1)
        self.Chrom = np.concatenate([Chrom1, Chrom2], axis=0)
        return self.Chrom

    def mut(self):
        # mutation
        Chrom = self.Chrom
        Pm = self.Pm
        pop = self.pop
        total_Lind = self.total_Lind
        mask = (np.random.rand(pop, total_Lind) < Pm) * 1
        self.Chrom = (mask + Chrom) % 2
        return self.Chrom

    def fit(self):
        max_iter = self.max_iter
        func = self.func
        for i in range(max_iter):
            self.crossover()
            self.mut()
            X = self.bs2rv()  # func的输入
            FitV = self.ranking()  # func的输出，ndarray
            sel_index = self.selection(FitV)  # index，选中的基因
            self.Chrom = self.Chrom[sel_index, :]  # 选出基因
            generation_best_X = X[FitV.argmax(), :]
            self.generation_best_X.append(generation_best_X)
            self.generation_best_ranking.append(FitV.max())
            self.FitV_history.append(FitV)
        general_best = self.generation_best_X[(np.array(self.generation_best_ranking)).argmax()]
        return general_best, func(general_best)

