## [scikit-opt](https://github.com/guofei9987/scikit-opt)
genetic algorithm, Particle swarm optimization, Simulated Annealing, Ant Colony Algorithm in Python

[中文文档](README_CN.md)

## 1. Genetic Algorithm

```py
from ga import GA


def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


ga = GA(func=demo_func, lb=[-1, -10, -5], ub=[2, 10, 2], max_iter=500)
best_x, best_y = ga.fit()
```
plot the result using matplotlib:
```py
import pandas as pd
import matplotlib.pyplot as plt
FitV_history = pd.DataFrame(ga.FitV_history)
fig, ax = plt.subplots(2, 1)
ax[0].plot(FitV_history.index, FitV_history.values, '.', color='red')
plt_max = FitV_history.max(axis=1)
ax[1].plot(plt_max.index, plt_max, label='max')
ax[1].plot(plt_max.index, plt_max.cummax())
plt.show()
```

![Figure_1-1](https://i.imgur.com/yT7lm8a.png)

### 1.1 Genetic Algorithm for TSP(Travelling Salesman Problem)
Just import the `GA_TSP`, it overloads the `crossover`, `mutation` to solve the TSP

Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
```py
import numpy as np

num_points = 8

points = range(num_points)
points_coordinate = np.random.rand(num_points, 2)
distance_matrix = np.zeros(shape=(num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        distance_matrix[i][j] = np.linalg.norm(points_coordinate[i] - points_coordinate[j], ord=2)
print('distance_matrix is: \n', distance_matrix)


def cal_total_distance(points):
    num_points, = points.shape
    total_distance = 0
    for i in range(num_points - 1):
        total_distance += distance_matrix[points[i], points[i + 1]]
    total_distance += distance_matrix[points[i + 1], points[0]]
    return total_distance
```

Do GA 
```py
from GA import GA_TSP
ga_tsp = GA_TSP(func=cal_total_distance, points=points, pop=50, max_iter=200, Pm=0.001)
best_points, best_distance = ga_tsp.fit()
```

Plot the result:
```py
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],'o-r')
plt.show()
```

![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ga_tsp.png?raw=true)


## 2. PSO


```py
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

pso = PSO(func=demo_func, dim=3)
fitness = pso.fit()
print('best_x is ',pso.gbest_x)
print('best_y is ',pso.gbest_y)
pso.plot_history()
```

![Figure_1-1](https://i.imgur.com/4C9Yjv7.png)


## 3. SA(Simulated Annealing)
```python
from SA import SA
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

sa = SA(func=demo_func, x0=[1, 1, 1])
x_star, y_star = sa.fit()
print(x_star, y_star)

```

```python
import matplotlib.pyplot as plt
import pandas as pd

plt.plot(pd.DataFrame(sa.f_list).cummin(axis=0))
plt.show()
```
![sa](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/sa.png?raw=true)

### 3.1 SA for TSP
Firstly, your data (the distance matrix). Here I generate the data randomly as a demo (find it in GA for TSP above)

DO SA for TSP
```python
from SA import SA_TSP
sa_tsp = SA_TSP(func=demo_func, x0=range(num_points))
best_points, best_distance = sa_tsp.fit()
```

plot the result
```python
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
```
![sa](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/sa_tsp.png?raw=true)

## 4. ASA for tsp (Ant Colony Algorithm)
ASA needs lots of parameter management, which is why I am not going to code it as a class.  

```bash
python ACA.py
```
![sa](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/aca_tsp.png?raw=true)


----------------------

[donate me](https://guofei9987.github.io/donate/)
