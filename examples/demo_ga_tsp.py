import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.demo_func import function_for_TSP

num_points, points_coordinate, distance_matrix, cal_total_distance = function_for_TSP(num_points=50)



# %%

from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, pop=500, max_iter=2000, Pm=0.3)
best_points, best_distance = ga_tsp.fit()

# %%
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()


#%%


# import matplotlib.pyplot as plt
#
# Y_history = ga_tsp.all_history_Y
# Y_history = pd.DataFrame(Y_history)
# fig, ax = plt.subplots(3, 1)
# ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
# plt_mean = Y_history.mean(axis=1)
# plt_max = Y_history.min(axis=1)
# ax[1].plot(plt_mean.index, plt_mean, label='mean')
# ax[1].plot(plt_max.index, plt_max, label='min')
# ax[1].set_title('mean and all Y of every generation')
# ax[1].legend()
#
# ax[2].plot(plt_max.index, plt_max.cummin())
# ax[2].set_title('best fitness of every generation')
# plt.show()
