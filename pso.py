import numpy as np
import matplotlib.pyplot as plt


class PSO():
    def __init__(self, func, pop, dim, max_iter):
        self.func = func
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pop = pop  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pop, self.dim))  # 所有粒子的位置
        self.V = np.zeros((self.pop, self.dim))  # 所有粒子的速度
        self.y = np.zeros(self.pop)
        self.pbest_x = np.zeros((self.pop, self.dim))  # 个体经历的最佳位置
        self.gbest_x = np.zeros((1, self.dim))  # 全局最佳位置
        self.pbest_y = np.zeros(self.pop)  # 每个个体的历史最佳适应值
        self.gbest_y = 1e10  # 全局最佳适应值
        self.gbest_y_hist = []  # 记录历史全局最优，用于画图

    def cal_y(self):
        # 计算y值
        y = []
        for i in self.X:
            y.append(self.func(i))
        self.y = np.array(y)

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
            self.V = self.w * self.V + self.c1 * self.r1 * (self.pbest_x - self.X) + self.c2 * self.r2 * (
            self.gbest_x - self.X)
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






