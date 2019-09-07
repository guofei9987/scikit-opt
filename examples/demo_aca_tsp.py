from sko.ACA import ACA_TSP
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(6)
num_points = 8

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
print('distance_matrix is: \n', distance_matrix)


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# test:
points = np.arange(num_points)  # generate index of points
cal_total_distance(points)

aca = ACA_TSP(func=cal_total_distance, n_dim=8,
              size_pop=10, max_iter=20,
              distance_matrix=distance_matrix)

best_x, best_y = aca.fit()

print(aca.y_best_history)
# %%

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
