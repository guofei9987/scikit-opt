import numpy as np
from sko.GA import GA, GA_TSP, ga_with_udf
from sko.GA import ranking_linear, ranking_raw, crossover_2point, selection_tournament, selection_roulette_2

options = {
    'ranking': {'udf': ranking_linear},
    'crossover': {'udf': crossover_2point},
    'selection': {'udf': selection_roulette_2}
}
GA_1 = ga_with_udf(GA, options)

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA_1(func=demo_func, n_dim=3, size_pop=100, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])
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

