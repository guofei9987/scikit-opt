# [scikit-opt](https://github.com/guofei9987/scikit-opt)

[![PyPI](https://img.shields.io/pypi/v/scikit-opt)](https://pypi.org/project/scikit-opt/)
[![release](https://img.shields.io/github/v/release/guofei9987/scikit-opt)](https://github.com/guofei9987/scikit-opt)
[![Build Status](https://travis-ci.com/guofei9987/scikit-opt.svg?branch=master)](https://travis-ci.com/guofei9987/scikit-opt)
[![codecov](https://codecov.io/gh/guofei9987/scikit-opt/branch/master/graph/badge.svg)](https://codecov.io/gh/guofei9987/scikit-opt)
[![Downloads](https://pepy.tech/badge/scikit-opt)](https://pepy.tech/project/scikit-opt)
[![Stars](https://img.shields.io/github/stars/guofei9987/scikit-opt?style=social)](https://github.com/guofei9987/scikit-opt/stargazers)
[![Forks](https://img.shields.io/github/forks/guofei9987/scikit-opt.svg?style=social)](https://github.com/guofei9987/scikit-opt/network/members)
[![Join the chat at https://gitter.im/guofei9987/scikit-opt](https://badges.gitter.im/guofei9987/scikit-opt.svg)](https://gitter.im/guofei9987/scikit-opt?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

一个封装了7种启发式算法的 Python 代码库  
（差分进化算法、遗传算法、粒子群算法、模拟退火算法、蚁群算法、鱼群算法、免疫优化算法）

# 安装

```bash
pip install scikit-opt
```
或者直接把源代码中的 `sko` 文件夹下载下来放本地也调用可以

# 特性
## 特性1：UDF（用户自定义算子）

举例来说，你想出一种新的“选择算子”，如下
-> Demo code: [examples/demo_ga_udf.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L1)
```python
# step1: define your own operator:
def selection_tournament(algorithm, tourn_size):
    FitV = algorithm.FitV
    sel_index = []
    for i in range(algorithm.size_pop):
        aspirants_index = np.random.choice(range(algorithm.size_pop), size=tourn_size)
        sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
    algorithm.Chrom = algorithm.Chrom[sel_index, :]  # next generation
    return algorithm.Chrom


```

导入包，并且创建遗传算法实例  
-> Demo code: [examples/demo_ga_udf.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L12)
```python
import numpy as np
from sko.GA import GA, GA_TSP

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + (x[2] - 0.5) ** 2
ga = GA(func=demo_func, n_dim=3, size_pop=100, max_iter=500, prob_mut=0.001,
        lb=[-1, -10, -5], ub=[2, 10, 2], precision=[1e-7, 1e-7, 1])

```
把你的算子注册到你创建好的遗传算法实例上  
-> Demo code: [examples/demo_ga_udf.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L20)
```python
ga.register(operator_name='selection', operator=selection_tournament, tourn_size=3)
```

scikit-opt 也提供了十几个算子供你调用  
-> Demo code: [examples/demo_ga_udf.py#s4](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L22)
```python
from sko.operators import ranking, selection, crossover, mutation

ga.register(operator_name='ranking', operator=ranking.ranking). \
    register(operator_name='crossover', operator=crossover.crossover_2point). \
    register(operator_name='mutation', operator=mutation.mutation)
```
做遗传算法运算
-> Demo code: [examples/demo_ga_udf.py#s5](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L28)
```python
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```

> 现在 **udf** 支持遗传算法的这几个算子：   `crossover`, `mutation`, `selection`, `ranking`

> Scikit-opt 也提供了十来个算子，参考[这里](https://github.com/guofei9987/scikit-opt/tree/master/sko/operators)

> 提供一个面向对象风格的自定义算子的方法，供进阶用户使用:

-> Demo code: [examples/demo_ga_udf.py#s6](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_udf.py#L31)
```python
class MyGA(GA):
    def selection(self, tourn_size=3):
        FitV = self.FitV
        sel_index = []
        for i in range(self.size_pop):
            aspirants_index = np.random.choice(range(self.size_pop), size=tourn_size)
            sel_index.append(max(aspirants_index, key=lambda i: FitV[i]))
        self.Chrom = self.Chrom[sel_index, :]  # next generation
        return self.Chrom

    ranking = ranking.ranking


demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + (x[2] - 0.5) ** 2
my_ga = MyGA(func=demo_func, n_dim=3, size_pop=100, max_iter=500, lb=[-1, -10, -5], ub=[2, 10, 2],
             precision=[1e-7, 1e-7, 1])
best_x, best_y = my_ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```

## 特性2：断点继续运行
例如，先跑10代，然后在此基础上再跑20代，可以这么写：
```python
from sko.GA import GA

func = lambda x: x[0] ** 2
ga = GA(func=func, n_dim=1)
ga.run(10)
ga.run(20)
```



## 特性3：4种加速方法
- [x] 矢量化计算：vectorization
- [x] 多线程计算：multithreading，适用于 IO 密集型目标函数
- [x] 多进程计算：multiprocessing，适用于 CPU 密集型目标函数
- [x] 缓存化计算：cached，适用于目标函数的每次输入有大量重复

see [https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py)

## 特性4: GPU 加速
GPU加速功能还比较简单，将会在 1.0.0 版本大大完善。  
有个 demo 已经可以在现版本运行了: [https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_gpu.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_gpu.py)


# 快速开始

## 1. 差分进化算法
**Step1**：定义你的问题，这个demo定义了有约束优化问题  
-> Demo code: [examples/demo_de.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_de.py#L1)
```python
'''
min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5
'''


def obj_func(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2


constraint_eq = [
    lambda x: 1 - x[1] - x[2]
]

constraint_ueq = [
    lambda x: 1 - x[0] * x[1],
    lambda x: x[0] * x[1] - 5
]

```

**Step2**: 做差分进化算法  
-> Demo code: [examples/demo_de.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_de.py#L25)
```python
from sko.DE import DE

de = DE(func=obj_func, n_dim=3, size_pop=50, max_iter=800, lb=[0, 0, 0], ub=[5, 5, 5],
        constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

```


## 2. 遗传算法
**第一步**：定义你的问题  
-> Demo code: [examples/demo_ga.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py#L1)
```python
import numpy as np


def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


```

**第二步**：运行遗传算法  
-> Demo code: [examples/demo_ga.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py#L14)
```python
from sko.GA import GA

ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, prob_mut=0.001, lb=[-1, -1], ub=[1, 1], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```

*第三步**：用 matplotlib 画出结果  
-> Demo code: [examples/demo_ga.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py#L20)
```python
import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
```


![Figure_1-1](https://img1.github.io/heuristic_algorithm/ga_1.png)

### 2.2 遗传算法用于旅行商问题
`GA_TSP` 针对TSP问题重载了 `交叉(crossover)`、`变异(mutation)` 两个算子

**第一步**，定义问题。  
这里作为demo，随机生成距离矩阵. 实战中从真实数据源中读取。

-> Demo code: [examples/demo_ga_tsp.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_tsp.py#L1)
```python
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

num_points = 50

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


```

**第二步**，调用遗传算法进行求解  
-> Demo code: [examples/demo_ga_tsp.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_tsp.py#L19)
```python

from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
best_points, best_distance = ga_tsp.run()

```

**第三步**，画出结果：   
-> Demo code: [examples/demo_ga_tsp.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga_tsp.py#L26)
```python
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()
```

![GA_TPS](https://img1.github.io/heuristic_algorithm/ga_tsp.png)


## 3. 粒子群算法
(PSO, Particle swarm optimization)

### 3.1 粒子群算法
**第一步**，定义问题  
-> Demo code: [examples/demo_pso.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso.py#L1)
```python
def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


```
**第二步**，做粒子群算法  
-> Demo code: [examples/demo_pso.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso.py#L6)
```python
from sko.PSO import PSO

pso = PSO(func=demo_func, n_dim=3, pop=40, max_iter=150, lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

```
**第三步**，画出结果  
-> Demo code: [examples/demo_pso.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso.py#L13)
```python
import matplotlib.pyplot as plt

plt.plot(pso.gbest_y_hist)
plt.show()
```

![PSO_TPS](https://img1.github.io/heuristic_algorithm/pso.png)

![pso_ani](https://img1.github.io/heuristic_algorithm/pso.gif)  
↑**see [examples/demo_pso.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso_ani.py)**

### 3.2 带非线性约束的粒子群算法

### 3.2 PSO with nonlinear constraint

加入你的非线性约束是个圆内的面积 `(x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2<=0`  
这样写代码:
```python
constraint_ueq = (
    lambda x: (x[0] - 1) ** 2 + (x[1] - 0) ** 2 - 0.5 ** 2
    ,
)
pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=max_iter, lb=[-2, -2], ub=[2, 2]
          , constraint_ueq=constraint_ueq)
```
可以有多个非线性约束，向 `constraint_ueq` 加就行了。

## 4. 模拟退火算法
(SA, Simulated Annealing)
### 4.1 模拟退火算法用于多元函数优化
**第一步**：定义问题  
-> Demo code: [examples/demo_sa.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L1)
```python
demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2

```
**第二步**，运行模拟退火算法  
-> Demo code: [examples/demo_sa.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L3)
```python
from sko.SA import SA

sa = SA(func=demo_func, x0=[1, 1, 1], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
best_x, best_y = sa.run()
print('best_x:', best_x, 'best_y', best_y)

```
![sa](https://img1.github.io/heuristic_algorithm/sa.png)

**第三步**，画出结果
-> Demo code: [examples/demo_sa.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa.py#L10)
```python
import matplotlib.pyplot as plt
import pandas as pd

plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
plt.show()

```

另外，scikit-opt 还提供了三种模拟退火流派: Fast, Boltzmann, Cauchy. 更多参见 [more sa](https://scikit-opt.github.io/scikit-opt/#/zh/more_sa)

### 4.2 模拟退火算法解决TSP问题（旅行商问题）  
**第一步**，定义问题。（我猜你已经无聊了，所以不黏贴这一步了）  

**第二步**，调用模拟退火算法  
-> Demo code: [examples/demo_sa_tsp.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa_tsp.py#L21)
```python
from sko.SA import SA_TSP

sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=10 * num_points)

best_points, best_distance = sa_tsp.run()
print(best_points, best_distance, cal_total_distance(best_points))
```


**第三步**，画出结果
-> Demo code: [examples/demo_sa_tsp.py#s3](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa_tsp.py#L28)
```python
from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots(1, 2)

best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(sa_tsp.best_y_history)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Distance")
ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
           marker='o', markerfacecolor='b', color='c', linestyle='-')
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")
plt.show()

```
![sa](https://img1.github.io/heuristic_algorithm/sa_tsp.png)

咱还有个动画  
![sa](https://img1.github.io/heuristic_algorithm/sa_tsp1.gif)  
↑**参考代码 [examples/demo_sa_tsp.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_sa_tsp.py)**

## 5. 蚁群算法
蚁群算法(ACA, Ant Colony Algorithm)解决TSP问题

-> Demo code: [examples/demo_aca_tsp.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_aca_tsp.py#L17)
```python
from sko.ACA import ACA_TSP

aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=50, max_iter=200,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

```

![ACA](https://img1.github.io/heuristic_algorithm/aca_tsp.png)




## 6. 免疫优化算法
(immune algorithm, IA)
-> Demo code: [examples/demo_ia.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ia.py#L6)
```python

from sko.IA import IA_TSP

ia_tsp = IA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=500, max_iter=800, prob_mut=0.2,
                T=0.7, alpha=0.95)
best_points, best_distance = ia_tsp.run()
print('best routine:', best_points, 'best_distance:', best_distance)

```

![IA](https://img1.github.io/heuristic_algorithm/ia2.png)


## 7. 人工鱼群算法
人工鱼群算法(artificial fish swarm algorithm, AFSA)

-> Demo code: [examples/demo_afsa.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_afsa.py#L1)
```python
def func(x):
    x1, x2 = x
    return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2


from sko.AFSA import AFSA

afsa = AFSA(func, n_dim=2, size_pop=50, max_iter=300,
            max_try_num=100, step=0.5, visual=0.3,
            q=0.98, delta=0.5)
best_x, best_y = afsa.run()
print(best_x, best_y)
```

# 使用本项目的论文


- [Heinrich, K., Zschech, P., Janiesch, C., & Bonin, M. (2021). Process data properties matter: Introducing gated convolutional neural networks (GCNN) and key-value-predict attention networks (KVP) for next event prediction with deep learning. Decision Support Systems, 113494.](https://www.sciencedirect.com/science/article/pii/S016792362100004X)
- [Mahbub, R. (2020). Algorithms and Optimization Techniques for Solving TSP.](https://raiyanmahbub.com/images/Research_Paper.pdf)
- [Li, J., Chen, T., Lim, K., Chen, L., Khan, S. A., Xie, J., & Wang, X. (2019). Deep learning accelerated gold nanocluster synthesis. Advanced Intelligent Systems, 1(3), 1900029.](https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.201900029)
- [Lin, J., & Emam, Y. Data Augmentation with Policy Optimization.](http://104.131.144.199/papers/autoaugment_ppo.pdf)
