#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/11
# @Author  : github.com/guofei9987

import numpy as np


class ACA_TSP:
    def __init__(self, func, n_dim,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 alpha=1, beta=2, rho=0.1,
                 ):
        self.func = func
        self.n_dim = n_dim  # 城市数量
        self.size_pop = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))  # 避免除零错误

        self.Tau = np.ones((n_dim, n_dim))  # 信息素矩阵，每次迭代都会更新
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)  # 某一代每个蚂蚁的爬行路径
        self.y = None  # 某一代每个蚂蚁的爬行总距离
        self.generation_best_X, self.generation_best_Y = [], []  # 记录各代的最佳情况
        self.x_best_history, self.y_best_history = self.generation_best_X, self.generation_best_Y  # 历史原因，为了保持统一
        self.best_x, self.best_y = None, None

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):  # 对每次迭代
            prob_matrix = (self.Tau ** self.alpha) * (self.prob_matrix_distance) ** self.beta  # 转移概率，无须归一化。
            for j in range(self.size_pop):  # 对每个蚂蚁
                self.Table[j, 0] = 0  # start point，其实可以随机，但没什么区别
                for k in range(self.n_dim - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.Table[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # 在这些点中做选择
                    prob = prob_matrix[self.Table[j, k], allow_list]
                    prob = prob / prob.sum()  # 概率归一化
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]
                    self.Table[j, k + 1] = next_point

            # 计算距离
            y = np.array([self.func(i) for i in self.Table])

            # 顺便记录历史最好情况
            index_best = y.argmin()
            x_best, y_best = self.Table[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # 计算需要新涂抹的信息素
            delta_tau = np.zeros((self.n_dim, self.n_dim))
            for j in range(self.size_pop):  # 每个蚂蚁
                for k in range(self.n_dim - 1):  # 每个节点
                    n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                    delta_tau[n1, n2] += 1 / y[j]  # 涂抹的信息素
                n1, n2 = self.Table[j, self.n_dim - 1], self.Table[j, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
                delta_tau[n1, n2] += 1 / y[j]  # 涂抹信息素

            # 信息素飘散+信息素涂抹
            self.Tau = (1 - self.rho) * self.Tau + delta_tau

        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y

    fit = run
