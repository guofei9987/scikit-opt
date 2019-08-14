import numpy as np


class SA:
    def __init__(self, func, x0, T=100, T_min=1e-7, L=300, q=0.9):
        self.func = func
        self.x = np.array(x0)
        self.T = T  # 初始温度
        self.T_min = T_min  # 终止温度
        self.L = L  # 各温度下的迭代次数（链长）
        self.q = q  # 降温速率
        self.f_list = []  # 记录历史解
        self.x_star, self.f_star = None, None

    def new_x(self, x):
        return 0.2 * np.random.randn(len(x)) + x

    def fit(self):
        func = self.func
        T = self.T
        T_min = self.T_min
        L = self.L
        q = self.q
        x = self.x

        f1 = func(x)
        self.x_star, self.f_star = x, f1  # 全局最优
        while T > T_min:
            for i in range(L):
                # 随机扰动
                x2 = self.new_x(x)
                f2 = func(x2)

                # 加入到全局列表
                if f2 < self.f_star:
                    self.x_star, self.f_star = x2, f2
                self.f_list.append(f2)

                # Metropolis
                df = f2 - f1
                if df < 0 or np.exp(-df / T) > np.random.rand():
                    x, f1 = x2, f2

            T = T * q  # 降温
        return self.x_star, self.f_star


class SA_TSP(SA):
    def new_x(self, x):
        x=x.copy()
        n1, n2 = np.random.randint(0, len(x), 2)
        n1, n2 = min(n1, n2), max(n1, n2)
        x[n1], x[n2] = x[n2], x[n1]
        return x
