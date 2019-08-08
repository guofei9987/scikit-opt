from PSO import PSO

from test_func import sphere as obj_func


print('------------')
print('starting PSO...')

def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

pso = PSO(func=obj_func, dim=3)
fitness = pso.fit()
print('best_x is ',pso.gbest_x)
print('best_y is ',pso.gbest_y)
pso.plot_history()




#%%
print('-------------')
print('starting GA...')

from GA import GA


def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


ga = GA(func=demo_func, lb=[-1, -10, -5], ub=[2, 10, 2], max_iter=500)
best_x, best_y = ga.fit()

print('best_x:',best_x)
print('best_y:',best_y)


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


#%%

print('-------------')
print('starting GA for TSP problem ...')
from GA import GA_TSP
import numpy as np

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


ga_tsp = GA_TSP(func=demo_func, points=points, pop=50, max_iter=200, Pm=0.001)

best_points, best_distance = ga_tsp.fit()

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],'o-r')
plt.show()
