def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2



#%%
from PSO import PSO

from test_func import sphere as obj_func

print('------------')
print('starting PSO...')


pso = PSO(func=obj_func, dim=3)
fitness = pso.fit()
print('best_x is ', pso.gbest_x)
print('best_y is ', pso.gbest_y)
pso.plot_history()

# %%
print('-------------')
print('starting GA...')

from GA import GA



ga = GA(func=demo_func, lb=[-1, -10, -5], ub=[2, 10, 2], max_iter=500)
best_x, best_y = ga.fit()

print('best_x:', best_x)
print('best_y:', best_y)

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


# %%
from SA import SA


sa = SA(func=demo_func, x0=[1, 1, 1])
x_star, y_star = sa.fit()
print(x_star, y_star)

import matplotlib.pyplot as plt
import pandas as pd

plt.plot(pd.DataFrame(sa.f_list).cummin(axis=0))
plt.show()
