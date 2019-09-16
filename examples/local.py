import numpy as np
from sko.GA import GA, GA_TSP, ga_with_udf
from sko.GA import ranking_linear, ranking_raw, crossover_2point, selection_tournament

options = {'ranking': {'udf': ranking_linear},
           'crossover': {'udf': crossover_2point},
           'selection': {'udf': selection_tournament}
           }
GA_1 = ga_with_udf(GA, options)

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA_1(func=demo_func, n_dim=3, size_pop=100, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])
best_x, best_y = ga.fit()

print('best_x:', best_x, '\n', 'best_y:', best_y)

# %%
# generation_best_X = self.X[self.FitV.argmax(), :]
# self.generation_best_X.append(generation_best_X)
# self.generation_best_ranking.append(self.FitV.max())
# self.FitV_history.append(self.FitV)
# %%
# ga.FitV
# ga.X[ga.FitV.argmax(),:]


# a=np.array([1,3,2,4,5,6])
# np.argsort(np.argsort(a))
