import numpy as np
from sko.GA import GA, GA_TSP, ga_with_udf

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, size_pop=100, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])


# You may want to define your own operator:(自定义你的算子)
def selection_tournament(self, tourn_size):
    FitV = self.FitV
    sel_index = []
    for i in range(self.size_pop):
        aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom


# Or import the operators we already defined. As below: (或者导入已经写好的算子)

from sko.GA import ranking_linear, ranking_raw, crossover_2point, selection_roulette_2, mutation

#
ga.register(operator_name='ranking', operator=ranking_linear). \
    register(operator_name='crossover', operator=crossover_2point). \
    register(operator_name='mutation', operator=mutation). \
    register(operator_name='selection', operator=selection_tournament, tourn_size=3)

# best_x, best_y = ga.run()
best_x, best_y = ga.fit()
print('best_x:', best_x, '\n', 'best_y:', best_y)

# %%

import pandas as pd
import matplotlib.pyplot as plt

Y_history = ga.all_history_Y
Y_history = pd.DataFrame(Y_history)
fig, ax = plt.subplots(3, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
plt_mean = Y_history.mean(axis=1)
plt_max = Y_history.min(axis=1)
ax[1].plot(plt_mean.index, plt_mean, label='mean')
ax[1].plot(plt_max.index, plt_max, label='min')
ax[1].set_title('mean and all Y of every generation')
ax[1].legend()

ax[2].plot(plt_max.index, plt_max.cummin())
ax[2].set_title('best fitness of every generation')
plt.show()
