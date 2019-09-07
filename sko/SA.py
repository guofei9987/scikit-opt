import numpy as np


class SA:
    def __init__(self, func, x0, T=100, T_min=1e-7, L=300, q=0.9):
        self.func = func
        self.x = np.array(x0)  # initial solution
        self.T = T  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = L  # num of iteration under every temperature（Long of Chain）
        self.q = q  # cool down speed
        self.f_list = []  # history
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
        x = x.copy()
        n1, n2 = np.random.randint(0, len(x), 2)
        x[n1], x[n2] = x[n2], x[n1]
        return x


def sa_register_udf(udf_func_dict):
    class SAUdf(SA):
        pass

    for udf_name in udf_func_dict:
        if udf_name == 'new_x':
            SAUdf.new_x = udf_func_dict[udf_name]
    return SAUdf
