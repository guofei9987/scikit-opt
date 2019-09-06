[:clipboard: Document](https://scikit-opt.github.io/), [:clipboard: 文档](https://scikit-opt.github.io/),  


# [scikit-opt](https://github.com/guofei9987/scikit-opt)
Heuristic Algorithms in Python  
Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Algorithm in Python

# install
```bash
pip install scikit-opt
```

# News:

**UDF** (user defined function) will be available in the next release! (version 0.2)

For example, you just worked out a new type of `selection` function.  
Now, your `selection` function is like this:
```python
def selection_elite(self, FitV):
    '''
    A new selection strategy.
    This strategy makes the elite (defined as the best one for a generation) 
    100% survive the selection
    '''
    print('udf selection actived')

    FitV = (FitV - FitV.min()) / (FitV.max() - FitV.min() + 1e-10) + 0.2
    # the worst one should still has a chance to be selected
    # the elite(defined as the best one for a generation) must survive the selection
    elite_index = np.array([FitV.argmax()])

    # do Roulette to select the next generation
    sel_prob = FitV / FitV.sum()
    roulette_index = np.random.choice(range(self.size_pop), size=self.size_pop - 1, p=sel_prob)
    sel_index = np.concatenate([elite_index, roulette_index])
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom
```

Regist your selection to GA
```python
from sko.GA import register_udf
GA_1 = register_udf({'selection': selection_elite})
```

Now do GA as usual
```python
demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA_1(func=demo_func, n_dim=3, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])
best_x, best_y = ga.fit()
#
print('best_x:', best_x, '\n', 'best_y:', best_y)
```
>Until Now, the **udf** surport `crossover`, `mutation`, `selection`, `ranking` of GA




# demo
## 1. Genetic Algorithm

```python
from sko.GA import GA


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
ga_tsp = GA_TSP(func=cal_total_distance, points=points, pop=50, max_iter=200, Pm=0.001)
best_points, best_distance = ga_tsp.fit()
```

Plot the result:
```python
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
fitness = pso.fit()
print('best_x is ',pso.gbest_x)
print('best_y is ',pso.gbest_y)
pso.plot_history()
```


![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/pso.png?raw=true)

## 3. SA(Simulated Annealing)
```python
from sko.SA import SA
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
from sko.SA import SA_TSP
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
