<table border="0" width="10%">
  <tr>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/1.jpg?raw=true" height="80" width="82"></td>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/2.jpg?raw=true" height="80" width="82"></td>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/3.jpg?raw=true" height="80" width="82"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/4.jpg?raw=true" height="80" width="82"></td>
    <td><img src="https://img.shields.io/github/stars/guofei9987/scikit-opt.svg?style=social"></td>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/6.jpg?raw=true" height="82" width="82"></td>
  </tr>
   <tr>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/7.jpg?raw=true" height="82" width="82"></td>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/8.jpg?raw=true" height="82" width="82"></td>
    <td><img src="https://github.com/guofei9987/pictures_for_blog/blob/master/tmp/9.jpg?raw=true" height="82" width="82"></td>
  </tr>
</table>



# [scikit-opt](https://github.com/guofei9987/scikit-opt)

[![PyPI](https://img.shields.io/pypi/v/scikit-opt)](https://pypi.org/project/scikit-opt/)
[![release](https://img.shields.io/github/v/release/guofei9987/scikit-opt)](https://github.com/guofei9987/scikit-opt)
[![codecov](https://codecov.io/gh/guofei9987/scikit-opt/branch/master/graph/badge.svg)](https://codecov.io/gh/guofei9987/scikit-opt)
[![PyPI_downloads](https://img.shields.io/pypi/dm/scikit-opt)](https://pypi.org/project/scikit-opt/)
[![Build Status](https://travis-ci.com/guofei9987/scikit-opt.svg?branch=master)](https://travis-ci.com/guofei9987/scikit-opt)
[![Stars](https://img.shields.io/github/stars/guofei9987/scikit-opt?style=social)](https://github.com/guofei9987/scikit-opt/stargazers)
[![Forks](https://img.shields.io/github/forks/guofei9987/scikit-opt.svg?style=social)](https://github.com/guofei9987/scikit-opt/network/members)
[![Join the chat at https://gitter.im/guofei9987/scikit-opt](https://badges.gitter.im/guofei9987/scikit-opt.svg)](https://gitter.im/guofei9987/scikit-opt?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)



Heuristic Algorithms in Python  
(Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Algorithm, Immune Algorithm,Artificial Fish Swarm Algorithm in Python)  


- **Documentation:** [https://scikit-opt.github.io/scikit-opt/#/docs/en](https://scikit-opt.github.io/scikit-opt/#/docs/en),
- **文档：** [https://scikit-opt.github.io/scikit-opt/#/docs/zh](https://scikit-opt.github.io/scikit-opt/#/docs/zh)  
- **Source code:** [https://github.com/guofei9987/scikit-opt](https://github.com/guofei9987/scikit-opt)


# install
```bash
pip install scikit-opt
```

## News:
All algorithms will be available on ~~TensorFlow/Spark~~ **pytorch** on version 0.4, getting parallel performance.  
DE(Differential Evolution Algorithm) will be complete on version 0.5  
Have fun!


## feature: UDF

**UDF** (user defined function) is available now!

For example, you just worked out a new type of `selection` function.  
Now, your `selection` function is like this:
```python
def selection_tournament(self, tourn_size):
    FitV = self.FitV
    sel_index = []
    for i in range(self.size_pop):
        aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom
```

Regist your udf to GA (Here we provide some operators)
```python
from sko.GA import GA, GA_TSP
from sko.GA import ranking_linear, ranking_raw, crossover_2point, selection_roulette_2, mutation


demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, size_pop=100, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])

#
ga.register(operator_name='ranking', operator=ranking_linear). \
    register(operator_name='crossover', operator=crossover_2point). \
    register(operator_name='mutation', operator=mutation). \
    register(operator_name='selection', operator=selection_tournament, tourn_size=3)
```

Now do GA as usual
```python
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```
>Until Now, the **udf** surport `crossover`, `mutation`, `selection`, `ranking` of GA

> We provide a dozen of operators see [here](https://github.com/guofei9987/scikit-opt/blob/master/sko/GA.py)



# Quick start
## 1. Genetic Algorithm

```python
import numpy as np
from sko.GA import GA


def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)


ga = GA(func=schaffer, n_dim=3, size_pop=100, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```
plot the result using matplotlib:
```py
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
```

![Figure_1-1](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ga_1.png?raw=true)

### 1.1 Genetic Algorithm for TSP(Travelling Salesman Problem)
Just import the `GA_TSP`, it overloads the `crossover`, `mutation` to solve the TSP

Firstly, your data (the distance matrix). Here I generate the data randomly as a demo:
```python
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
```python
from sko.GA import GA_TSP
ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=300, max_iter=800, Pm=0.3)
best_points, best_distance = ga_tsp.run()
```

Plot the result:
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],'o-r')
plt.show()
```

![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ga_tsp.png?raw=true)


## 2. PSO(Particle swarm optimization)


```python
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

from sko.PSO import PSO
pso = PSO(func=demo_func, dim=3)
fitness = pso.run()
print('best_x is ',pso.gbest_x)
print('best_y is ',pso.gbest_y)
pso.plot_history()
```


![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/pso.png?raw=true)

## 3. SA(Simulated Annealing)
```python
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

from sko.SA import SA
sa = SA(func=demo_func, x0=[1, 1, 1])
x_star, y_star = sa.run()
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
from sko.SA import SA_TSP
sa_tsp = SA_TSP(func=demo_func, x0=range(num_points))
best_points, best_distance = sa_tsp.run()
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

## 4. ACA for tsp (Ant Colony Algorithm)


```bash
aca = ACA_TSP(func=cal_total_distance, n_dim=8,
              size_pop=10, max_iter=20,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()
```
![ACA](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/aca_tsp.png?raw=true)


## 5. immune algorithm (IA)

```python
from sko.IA import IA_TSP_g as IA_TSP

ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points, pop=500, max_iter=2000, Pm=0.2,
                T=0.7, alpha=0.95)
best_points, best_distance = ia_tsp.run()
print('best routine:', best_points, 'best_distance:', best_distance)
```

![IA](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ia2.png?raw=true)

## 6. artificial fish swarm algorithm (AFSA)

```python
def func(x):
    x1, x2 = x
    return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2


from sko.ASFA import ASFA

asfa = ASFA(func, n_dim=2, size_pop=50, max_iter=300,
            max_try_num=100, step=0.5, visual=0.3,
            q=0.98, delta=0.5)
best_x, best_y = asfa.run()
print(best_x, best_y)
```
