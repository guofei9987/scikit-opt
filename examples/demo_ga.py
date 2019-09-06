demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
from sko.GA import GA

ga = GA(func=demo_func,n_dim=3, lb=[-1, -10, -5], ub=[2, 10, 2], max_iter=500)
best_x, best_y = ga.fit()

print('best_x:', best_x, '\n', 'best_y:', best_y)

import pandas as pd
import matplotlib.pyplot as plt

FitV_history = ga.FitV_history
FitV_history = pd.DataFrame(FitV_history)
fig, ax = plt.subplots(3, 1)
ax[0].plot(FitV_history.index, FitV_history.values, '.', color='red')
plt_mean = FitV_history.mean(axis=1)
plt_max = FitV_history.max(axis=1)
ax[1].plot(plt_mean.index, plt_mean, label='mean')
ax[1].plot(plt_max.index, plt_max, label='max')
ax[1].set_title('mean and all fitness of every generation')
ax[1].legend()

ax[2].plot(plt_max.index, plt_max.cummax())
ax[2].set_title('best fitness of every generation')
plt.show()


# %%
