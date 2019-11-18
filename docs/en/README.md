# [scikit-opt](https://github.com/guofei9987/scikit-opt)

[![PyPI](https://img.shields.io/pypi/v/scikit-opt)](https://pypi.org/project/scikit-opt/)
[![release](https://img.shields.io/github/v/release/guofei9987/scikit-opt)](https://github.com/guofei9987/scikit-opt)
[![Build Status](https://travis-ci.com/guofei9987/scikit-opt.svg?branch=master)](https://travis-ci.com/guofei9987/scikit-opt)
[![codecov](https://codecov.io/gh/guofei9987/scikit-opt/branch/master/graph/badge.svg)](https://codecov.io/gh/guofei9987/scikit-opt)
[![PyPI_downloads](https://img.shields.io/pypi/dm/scikit-opt)](https://pypi.org/project/scikit-opt/)
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
-> Demo code: [examples/demo_ga_udf.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L1)
```python
# step1: define your own operator:
def selection_tournament(self, tourn_size):
    FitV = self.FitV
    sel_index = []
    for i in range(self.size_pop):
        aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    self.Chrom = self.Chrom[sel_index, :]  # next generation
    return self.Chrom


```

Import and build ga  
-> Demo code: [examples/demo_ga_udf.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L12)
```python
import numpy as np
from sko.GA import GA, GA_TSP

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, size_pop=100, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])

```
Regist your udf to GA  
-> Demo code: [examples/demo_ga_udf.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L19)
```python
ga.register(operator_name='selection', operator=selection_tournament, tourn_size=3)
```

scikit-opt also provide some operators  
-> Demo code: [examples/demo_ga_udf.py#s4](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L21)
```python
from sko.GA import ranking_linear, ranking_raw, crossover_2point, selection_roulette_2, mutation

ga.register(operator_name='ranking', operator=ranking_linear). \
    register(operator_name='crossover', operator=crossover_2point). \
    register(operator_name='mutation', operator=mutation)

```
Now do GA as usual  
-> Demo code: [examples/demo_ga_udf.py#s5](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L28)
```python
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

```

> Until Now, the **udf** surport `crossover`, `mutation`, `selection`, `ranking` of GA

> scikit-opt provide a dozen of operators, see [here](https://github.com/guofei9987/scikit-opt/blob/master/sko/GA.py)



# Quick start
## 1. Genetic Algorithm
-> Demo code: [examples/demo_ga.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py#L1)
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


ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

```
plot the result using matplotlib  
-> Demo code: [examples/demo_ga.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py#L19)
```python
import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
```

![Figure_1-1](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ga_1.png?raw=true)

### 1.1 Genetic Algorithm for TSP(Travelling Salesman Problem)
Just import the `GA_TSP`, it overloads the `crossover`, `mutation` to solve the TSP

Firstly, prepare your data (the distance matrix). Here I generate the data randomly as a demo:  
-> Demo code: [examples/demo_ga_tsp.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_tsp.py#L1)
```python
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

num_points = 8

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


```

Do GA  
-> Demo code: [examples/demo_ga_tsp.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_tsp.py#L19)
```python

from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=300, max_iter=800, Pm=0.3)
best_points, best_distance = ga_tsp.run()

```

Plot the result:  
-> Demo code: [examples/demo_ga_tsp.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_tsp.py#L26)
```python
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
```

![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ga_tsp.png?raw=true)


## 2. PSO(Particle swarm optimization)

### 2.1 PSO with constraint
-> Demo code: [examples/demo_pso.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso.py#L1)
```python
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


from sko.PSO import PSO

pso = PSO(func=demo_func, dim=3, lb=[0, -1, 0.5], ub=[1, 1, 1])
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

import matplotlib.pyplot as plt

plt.plot(pso.gbest_y_hist)
plt.show()

```


![PSO_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/pso.png?raw=true)

### 2.2 PSO without constraint
-> Demo code: [examples/demo_pso.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso.py#L17)
```python
pso = PSO(func=demo_func, dim=3)
fitness = pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
```

## 3. SA(Simulated Annealing)
### 3.1 SA for multiple function
-> Demo code: [examples/demo_sa.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L1)
```python
from sko.SA import SA

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
sa = SA(func=demo_func, x0=[1, 1, 1])
x_star, y_star = sa.run()
print(x_star, y_star)

```
-> Demo code: [examples/demo_sa.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L8)
```python
import matplotlib.pyplot as plt
import pandas as pd

plt.plot(pd.DataFrame(sa.f_list).cummin(axis=0))
plt.show()
```
![sa](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/sa.png?raw=true)

### 3.2 SA for TSP
Firstly, prepare your data (the distance matrix). See GA_TSP.  

DO SA for TSP  
-> Demo code: [examples/demo_sa_tsp.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa_tsp.py#L19)
```python
from sko.SA import SA_TSP

sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points))

best_points, best_distance = sa_tsp.fit()
print(best_points, best_distance, cal_total_distance(best_points))
```

plot the result  
-> Demo code: [examples/demo_sa_tsp.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa_tsp.py#L19)
```python
from sko.SA import SA_TSP

sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points))

best_points, best_distance = sa_tsp.fit()
print(best_points, best_distance, cal_total_distance(best_points))
```
![sa](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/sa_tsp.png?raw=true)

### 3.2 SA for real function optimization
-> Demo code: [examples/demo_sa.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L1)
```python
from sko.SA import SA

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
sa = SA(func=demo_func, x0=[1, 1, 1])
x_star, y_star = sa.run()
print(x_star, y_star)

```



## 4. ACA (Ant Colony Algorithm) for tsp 
-> Demo code: [examples/demo_aca_tsp.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_aca_tsp.py#L23)
```python
from sko.ACA import ACA_TSP

aca = ACA_TSP(func=cal_total_distance, n_dim=8,
              size_pop=10, max_iter=20,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

```

![ACA](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/aca_tsp.png?raw=true)


## 5. immune algorithm (IA)
-> Demo code: [examples/demo_ia.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ia.py#L6)
```python

from sko.IA import IA_TSP

ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points, pop=500, max_iter=2000, Pm=0.2,
                T=0.7, alpha=0.95)
best_points, best_distance = ia_tsp.run()
print('best routine:', best_points, 'best_distance:', best_distance)

```

![IA](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ia2.png?raw=true)

## 6. artificial fish swarm algorithm (AFSA)
-> Demo code: [examples/demo_asfs.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_asfs.py#L1)
```python
def func(x):
    x1, x2 = x
    return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2


from sko.ASFA import ASFA

asfa = ASFA(func, n_dim=2, size_pop=50, max_iter=300,
            max_try_num=100, step=0.5, visual=0.3,
            q=0.98, delta=0.5)
best_x, best_y = asfa.fit()
print(best_x, best_y)
```
