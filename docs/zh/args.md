
## 入参一览

可以使用类似 `help(GA)`, `GA?` 查看详细介绍，例如：
```python
import sko

help(sko.GA.GA)
help(sko.GA.GA_TSP)
help(sko.PSO.PSO)
help(sko.DE.DE)
help(sko.SA.SA)
help(sko.SA.SA_TSP)
help(sko.ACA.ACA_TSP)
help(sko.IA.IA_TSP)
help(sko.AFSA.AFSA)
```

### GA

| 入参              | 默认值    | 意义                     |
|-----------------|--------|------------------------|
| func            | \-     | 目标函数                   |
| n\_dim          | \-     | 目标函数的维度                |
| size\_pop       | 50     | 种群规模                   |
| max\_iter       | 200    | 最大迭代次数                 |
| prob\_mut       | 0\.001 | 变异概率                   |
| lb              | \-1    | 每个参数的最小值               |
| ub              | 1      | 每个参数的最大值               |
| constraint\_eq  | 空元组    | 线性约束                   |
| constraint\_ueq | 空元组    | 非线性约束                  |
| precision       | 1e\-7  | 精准度，int/float或者它们组成的列表 |


### GA_TSP

| 入参        | 默认值    | 意义     |
|-----------|--------|--------|
| func      | \-     | 目标函数   |
| n\_dim    | \-     | 城市个数   |
| size\_pop | 50     | 种群规模   |
| max\_iter | 200    | 最大迭代次数 |
| prob\_mut | 0\.001 | 变异概率   |


### PSO

| 入参        | 默认值  | 意义       |
|-----------|------|----------|
| func      | \-   | 目标函数     |
| n\_dim    | \-   | 目标函数的维度  |
| size\_pop | 50   | 种群规模     |
| max\_iter | 200  | 最大迭代次数   |
| lb        | None | 每个参数的最小值 |
| ub        | None | 每个参数的最大值 |
| w         | 0\.8 | 惯性权重     |
| c1        | 0\.5 | 个体记忆     |
| c2        | 0\.5 | 集体记忆     |
| constraint\_ueq | 空元组    | 非线性约束                  |

### DE

| 入参        | 默认值    | 意义       |
|-----------|--------|----------|
| func      | \-     | 目标函数     |
| n\_dim    | \-     | 目标函数的维度  |
| size\_pop | 50     | 种群规模     |
| max\_iter | 200    | 最大迭代次数   |
| prob\_mut | 0\.001 | 变异概率     |
| F         | 0\.5   | 变异系数     |
| lb        | \-1    | 每个参数的最小值 |
| ub        | 1      | 每个参数的最大值 |
| constraint\_eq  | 空元组    | 线性约束                   |
| constraint\_ueq | 空元组    | 非线性约束                  |

### SA

| 入参                 | 默认值   | 意义    |
|--------------------|-------|-------|
| func               | \-    | 目标函数  |
| x0                 | \-    | 迭代初始点 |
| T\_max             | 100   | 最大温度  |
| T\_min             | 1e\-7 | 最小温度  |
| L                  | 300   | 链长    |
| max\_stay\_counter | 150   | 冷却耗时  |


### ACA_TSP

| 入参               | 默认值  | 意义                   |
|------------------|------|----------------------|
| func             | \-   | 目标函数                 |
| n\_dim           | \-   | 城市个数                 |
| size\_pop        | 10   | 蚂蚁数量                 |
| max\_iter        | 20   | 最大迭代次数               |
| distance\_matrix | \-   | 城市之间的距离矩阵，用于计算信息素的挥发 |
| alpha            | 1    | 信息素重要程度              |
| beta             | 2    | 适应度的重要程度             |
| rho              | 0\.1 | 信息素挥发速度              |


### IA_TSP

| 入参        | 默认值    | 意义                               |
|-----------|--------|----------------------------------|
| func      | \-     | 目标函数                             |
| n\_dim    | \-     | 城市个数                             |
| size\_pop | 50     | 种群规模                             |
| max\_iter | 200    | 最大迭代次数                           |
| prob\_mut | 0\.001 | 变异概率                             |
| T         | 0\.7   | 抗体与抗体之间的亲和度阈值，大于这个阈值认为亲和，否则认为不亲和 |
| alpha     | 0\.95  | 多样性评价指数，也就是抗体和抗原的重要性/抗体浓度重要性     |


### AFSA

| 入参            | 默认值   | 意义               |
|---------------|-------|------------------|
| func          | \-    | 目标函数             |
| n\_dim        | \-    | 目标函数的维度          |
| size\_pop     | 50    | 种群规模             |
| max\_iter     | 300   | 最大迭代次数           |
| max\_try\_num | 100   | 最大尝试捕食次数         |
| step          | 0\.5  | 每一步的最大位移比例       |
| visual        | 0\.3  | 鱼的最大感知范围         |
| q             | 0\.98 | 鱼的感知范围衰减系数       |
| delta         | 0\.5  | 拥挤度阈值，越大越容易聚群和追尾 |

## 输出一览


### GA&GA_TSP

- `ga.generation_best_Y` 每一代的最优函数值
- `ga.generation_best_X` 每一代的最优函数值对应的输入值
- `ga.all_history_FitV` 每一代的每个个体的适应度
- `ga.all_history_Y` 每一代每个个体的函数值
- `ga.best_y` 最优函数值
- `ga.best_x` 最优函数值对应的输入值

### DE


- `de.generation_best_Y` 每一代的最优函数值
- `de.generation_best_X` 每一代的最优函数值对应的输入值
- `de.all_history_Y` 每一代每个个体的函数值
- `de.best_y` 最优函数值
- `de.best_x` 最优函数值对应的输入值


### PSO
- `pso.record_value` 每一代的粒子位置、粒子速度、对应的函数值。`pso.record_mode = True` 才开启记录
- `pso.gbest_y_hist` 历史最优函数值
- `pso.best_y` 最优函数值 （迭代中使用的是 `pso.gbest_x`, `pso.gbest_y`）
- `pso.best_x` 最优函数值对应的输入值



### SA

- `de.generation_best_Y` 每一代的最优函数值
- `de.generation_best_X` 每一代的最优函数值对应的输入值
- `sa.best_x` 最优函数值
- `sa.best_y` 最优函数值对应的输入值

### ACA
- `de.generation_best_Y` 每一代的最优函数值
- `de.generation_best_X` 每一代的最优函数值对应的输入值
- `aca.best_y` 最优函数值
- `aca.best_x` 最优函数值对应的输入值

### AFSA
- `afsa.best_x` 最优函数值
- `afsa.best_y` 最优函数值对应的输入值

### IA

- `ia.generation_best_Y` 每一代的最优函数值
- `ia.generation_best_X` 每一代的最优函数值对应的输入值
- `ia.all_history_FitV` 每一代的每个个体的适应度
- `ia.all_history_Y` 每一代每个个体的函数值
- `ia.best_y` 最优函数值
- `ia.best_x` 最优函数值对应的输入值
