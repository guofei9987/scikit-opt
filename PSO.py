import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, func, dim, pop=40, max_iter=150):
        self.func = func
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pop, self.dim))  # location of particles, which is the value of variables of func
        self.V = np.zeros((self.pop, self.dim))  # speed of particles
        self.y = np.zeros(self.pop)
        self.pbest_x = np.zeros((self.pop, self.dim))  # 个体经历的最佳位置
        self.gbest_x = np.zeros((1, self.dim))  # 全局最佳位置
        self.pbest_y = np.zeros(self.pop)  # 每个个体的历史最佳适应值
        self.gbest_y = np.inf  # 全局最佳适应值
        self.gbest_y_hist = []  # 记录历史全局最优，用于画图

    def cal_y(self):
        # calculate y
        self.y = np.array([self.func(i) for i in self.X])

    def init_Population(self):
        X = np.random.rand(self.pop, self.dim)
        V = np.random.rand(self.pop, self.dim)
        self.X = X
        self.V = V
        self.cal_y()
        self.pbest_y = self.y
        gbest_index = self.pbest_y.argmax()
        self.gbest_x = X[gbest_index, :]
        self.gbest_y = self.pbest_y[gbest_index]
        self.pbest_x = X.copy()

    def fit(self):
        self.init_Population()
        for i in range(self.pop):
            self.V = self.w * self.V + \
                     self.c1 * self.r1 * (self.pbest_x - self.X) + \
                     self.c2 * self.r2 * (self.gbest_x - self.X)
            self.X = self.X + self.V
            self.cal_y()

            for i in range(self.pop):
                if self.pbest_y[i] > self.y[i]:
                    self.pbest_x[i, :] = self.X[i, :]
                    self.pbest_y[i] = self.y[i]
            if self.gbest_y > self.y.min():
                self.gbest_x = self.X[self.y.argmin(), :]
                self.gbest_y = self.y.min()
            self.gbest_y_hist.append(self.gbest_y)

    def plot_history(self):
        plt.plot(self.gbest_y_hist)
        plt.show()
