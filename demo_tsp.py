import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_points = 8

points = range(num_points)
points_coordinate = np.random.rand(num_points, 2)
distance_matrix = np.zeros(shape=(num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        distance_matrix[i][j] = np.linalg.norm(points_coordinate[i] - points_coordinate[j], ord=2)
print('distance_matrix is: \n', distance_matrix)


def demo_func(points):
    num_points, = points.shape
    total_distance = 0
    for i in range(num_points - 1):
        total_distance += distance_matrix[points[i], points[i + 1]]
    total_distance += distance_matrix[points[i + 1], points[0]]
    return total_distance

#%%
print('-------------')
print('starting GA for TSP problem ...')
from GA import GA_TSP
ga_tsp = GA_TSP(func=demo_func, points=points, pop=50, max_iter=200, Pm=0.001)

best_points, best_distance = ga_tsp.fit()

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()


#%%

print('-------------')
print('starting SA for TSP problem ...')
from SA import SA_TSP
sa_tsp = SA_TSP(func=demo_func, x0=range(num_points))

best_points, best_distance = sa_tsp.fit()
print(best_points, best_distance,demo_func(best_points))

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()