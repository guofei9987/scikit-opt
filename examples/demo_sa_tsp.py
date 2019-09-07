import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt

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


# %%

print('-------------')
print('starting SA for TSP problem ...')
from sko.SA import SA_TSP

sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points))

best_points, best_distance = sa_tsp.fit()
print(best_points, best_distance, cal_total_distance(best_points))

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
