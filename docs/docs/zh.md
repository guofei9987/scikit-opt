## 0.0 代码仓库


[![release](https://img.shields.io/github/v/release/guofei9987/scikit-opt)](https://github.com/guofei9987/scikit-opt)
[![Stars](https://img.shields.io/github/stars/guofei9987/scikit-opt.svg?label=Stars&style=social)](https://github.com/guofei9987/scikit-opt/stargazers)
[![Forks](https://img.shields.io/github/forks/guofei9987/scikit-opt.svg?label=Fork&style=social)](https://github.com/guofei9987/scikit-opt/network/members)

## 0.1安装

```bash
pip install scikit-opt
```
或者直接把源代码中的 `sko` 文件夹下载下来放本地也调用可以

## 0.2 第一个遗传算法
```python
demo_func=lambda x: x[0]**2 + x[1]**2 + x[2]**2
ga = GA(func=demo_func,n_dim=3, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```
恭喜，你已经跑完了第一个遗传算法！

## 0.3 自定义算子UDF
**UDF** (用户自定义算子, user defined function) 会在0.2版本可用。  

例如，如果你构造了一种 `选择算子`(`selection`)，你的算子是这样的：  
  
改进的
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

把你的 UDF 自定义算子注册到遗传算法对象上：（这里为了还展示了我们提供的算子的注册）
```python
from sko.GA import GA, GA_TSP
from sko.GA import ranking_linear, ranking_raw, crossover_2point, selection_roulette_2, mutation


demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, size_pop=100, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2])


ga.register(operator_name='ranking', operator=ranking_linear). \
    register(operator_name='crossover', operator=crossover_2point). \
    register(operator_name='mutation', operator=mutation). \
    register(operator_name='selection', operator=selection_tournament, tourn_size=3)
```

像往常一样运行遗传算法：
```python
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```
恭喜你，成功了。  


> 现在 **udf** 支持遗传算法的这几个算子：   `crossover`, `mutation`, `selection`, `ranking`

> 提供了十来个算子 参考[这里](https://github.com/guofei9987/scikit-opt/blob/master/sko/GA.py)



## 1. 遗传算法(Genetic Algorithm)
### 1.1 遗传算法用于多元实函数优化

定义目标函数
```python
import numpy as np


def schaffer(p):
    '''
    此函数具有无数个极小值点、强烈的震荡形态，所以很难找到全局最优值
    在(0,0)处取的最值0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)


```

调入遗传算法求解器
```python
from sko.GA import GA
ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```

用 matplotlib 画出结果
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

### 1.2 遗传算法用于旅行商问题
`GA_TSP` 针对TSP问题重载了 `交叉(crossover)`、`变异(mutation)` 两个算子

这里作为demo，随机生成距离矩阵. 实战中从真实数据源中读取。

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
from sko.GA import GA_TSP
ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=300, max_iter=800, Pm=0.3)
best_points, best_distance = ga_tsp.run()
```

画出结果
```py
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],'o-r')
plt.show()
```

![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ga_tsp.png?raw=true)

## 2. 粒子群算法

粒子群算法(PSO, Particle swarm optimization)

定义目标函数
```py
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
```

优化求解
```python
from sko.PSO import PSO
pso = PSO(func=demo_func, dim=3)
fitness = pso.run()
print('best_x is ',pso.gbest_x)
print('best_y is ',pso.gbest_y)
pso.plot_history()
```

![GA_TPS](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/pso.png?raw=true)



## 3. 模拟退火算法
模拟退火算法(SA, Simulated Annealing)
### 3.1 模拟退火算法用于多元函数优化

```python
from sko.SA import SA
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

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

### 3.2 模拟退火算法解决TSP问题
TSP问题（旅行商问题）  


作为demo，生成模拟数据（代码与遗传算法解决TSP问题一样，这里省略）

调用模拟退火算法
```python
from sko.SA import SA_TSP
sa_tsp = SA_TSP(func=demo_func, x0=range(num_points))
best_points, best_distance = sa_tsp.run()
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




## 4. 蚁群算法解决TSP问题
蚁群算法(ACA, Ant Colony Algorithm)  

```bash
aca = ACA_TSP(func=cal_total_distance, n_dim=8,
              size_pop=10, max_iter=20,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()
```
![aca_tsp](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/aca_tsp.png?raw=true)



## 5. 免疫优化算法(immune algorithm, IA)

```python
from sko.IA import IA_TSP_g as IA_TSP

ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points, pop=500, max_iter=2000, Pm=0.2,
                T=0.7, alpha=0.95)
best_points, best_distance = ia_tsp.run()
print('best routine:', best_points, 'best_distance:', best_distance)
```

![ia](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/ia2.png?raw=true)

## 6. 人工鱼群算法(artificial fish swarm algorithm, AFSA)

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

# Q&A
## 如何进行整数规划

在多维优化时，想让哪个变量限制为整数，就设定 `precision` 为 1即可。  
例如，我想让我的自定义函数 `demo_func` 的第一个变量限制为整数，那么久设定 `precision` 的第一个数为1，例子如下：
```python
from sko.GA import GA

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=[0, 0, 0], ub=[1, 1, 1], precision=[1, 1e-7, 1e-7])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```