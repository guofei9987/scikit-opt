
动动手点个 star，让更多人能用到好的 Python 库

-----------------------------------
## [scikit-opt](https://github.com/guofei9987/scikit-opt)  

遗传算法、粒子群算法、模拟退火、蚁群算法、免疫优化算法、鱼群算法较好的封装  

[English Document](README.md)

## 1. 遗传算法(Genetic Algorithm)

定义目标函数
```python
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
```

调入遗传算法求解器
```python
from ga import GA
ga = GA(func=demo_func, lb=[-1, -10, -5], ub=[2, 10, 2], max_iter=500)
best_x, best_y = ga.fit()
```

用 matplotlib 画出结果
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

### 1.1 用遗传算法解决TSP问题（旅行商问题）
`GA_TSP` 针对TSP问题重载了 `交叉(crossover)`、`变异(mutation)` 两个算子

这里作为demo，随机生成距离矩阵. 实战中，从数据源中读取。

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

然后调用遗传算法进行求解
```py
from GA import GA_TSP
ga_tsp = GA_TSP(func=cal_total_distance, points=points, pop=50, max_iter=200, Pm=0.001)
best_points, best_distance = ga_tsp.fit()
```

画出结果
```py
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],'o-r')
plt.show()
```

![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ga_tsp.png?raw=true)


## 2. 粒子群算法(PSO)

定义目标函数
```py
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
```

优化求解
```python
pso = PSO(func=demo_func, dim=3)
fitness = pso.fit()
print('best_x is ',pso.gbest_x)
print('best_y is ',pso.gbest_y)
pso.plot_history()
```

![Figure_1-1](https://i.imgur.com/4C9Yjv7.png)


## 3. 模拟退火算法(SA, Simulated Annealing)
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

### 3.1 模拟退火算法解决TSP问题（旅行商问题）

作为demo，生成模拟数据（代码与遗传算法解决TSP问题一样，这里省略）

调用模拟退火算法
```python
from SA import SA_TSP
sa_tsp = SA_TSP(func=demo_func, x0=range(num_points))
best_points, best_distance = sa_tsp.fit()
```

画出结果
```python
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
```
![sa](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/sa_tsp.png?raw=true)

## 4. 蚁群算法(ACA, Ant Colony Algorithm)解决TSP问题
蚁群算法在超参数调试上需要大量工作，所以没有深度封装
```bash
python ACA.py
```
![sa](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/aca_tsp.png?raw=true)


----------------------

[donate me](https://guofei9987.github.io/donate/)
