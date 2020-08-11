
## 遗传算法进行整数规划

在多维优化时，想让哪个变量限制为整数，就设定 `precision` 为 **整数** 即可。  
例如，我想让我的自定义函数 `demo_func` 的某些变量限制为整数+浮点数（分别是隔2个，隔1个，浮点数），那么就设定 `precision=[2, 1, 1e-7]`  
例子如下：
```python
from sko.GA import GA

demo_func = lambda x: (x[0] - 1) ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=[-1, -1, -1], ub=[5, 1, 1], precision=[2, 1, 1e-7])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```

说明：
- 当 `precision` 为整数时，对应的自变量会启用整数规划模式。
- 在整数规划模式下，变量的取值可能个数最好是 $2^n$，这样收敛速度快，效果好。
<!-- - 在整数规划模式下，如果某个变量的取值可能个数不是 $2^n$，`GA` 会做这些事：
    1. 调整 `ub`，使得可能取值扩展成 $2^n$ 个
    2. 增加一个 **不等式约束** `constraint_ueq`，并使用罚函数法来处理
    3. 如果你的 **等式约束** `constraint_eq` 和 **不等式约束** `constraint_ueq` 已经很多了，更加推荐先手动做调整，以规避可能个数不是 $2^n$这种情况，毕竟太多的约束会影响性能。 -->
- 如果 `precision` 不是整数（例如是0.5）,则不会进入整数规划模式，如果还想用这个模式，那么把对应自变量乘以2，这样 `precision` 就是整数了。

## 遗传TSP问题如何固定起点和终点？
固定起点和终点要求路径不闭合（因为如果路径是闭合的，固定与不固定结果实际上是一样的）  

假设你的起点和终点坐标指定为(0, 0) 和 (1, 1)，这样构建目标函数
- 起点和终点不参与优化。假设共有n+2个点，优化对象是中间n个点
- 目标函数（总距离）按实际去写。


```python
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

num_points = 20

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
start_point=[[0,0]]
end_point=[[1,1]]
points_coordinate=np.concatenate([points_coordinate,start_point,end_point])
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    # start_point,end_point 本身不参与优化。给一个固定的值，参与计算总路径
    routine = np.concatenate([[num_points], routine, [num_points+1]])
    return sum([distance_matrix[routine[i], routine[i + 1]] for i in range(num_points+2-1)])
```

正常运行并画图：
```python
from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
best_points, best_distance = ga_tsp.run()


fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([[num_points],best_points, [num_points+1]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()
```

![image](https://user-images.githubusercontent.com/19920283/83831463-0ac6a400-a71a-11ea-8692-beac5f465111.png)

更多说明，[这里](https://github.com/guofei9987/scikit-opt/issues/58)

## 如何设定初始点或初始种群

- 对于遗传算法 `GA`, 运行 `ga=GA(**params)` 生成模型后，赋值设定初始种群，例如 `ga.Chrom = np.random.randint(0,2,size=(80,20))`
- 对于差分进化算法 `DE`，设定 `de.X` 为初始 X.  
- 对于模拟退火算法 `SA`，入参 `x0` 就是初始点.
- 对于粒子群算法 `PSO`，手动赋值 `pso.X` 为初始 X, 然后执行 `pso.cal_y(); pso.update_gbest(); pso.update_pbest()` 来更新历史最优点
