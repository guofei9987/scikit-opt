import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sko.demo_func import function_for_TSP

num_points, points_coordinate, distance_matrix, cal_total_distance = function_for_TSP(num_points=15)



# %%

from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, pop=200, max_iter=1000, Pm=0.3)
best_points, best_distance = ga_tsp.fit()

# %%
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
