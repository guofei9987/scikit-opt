## 目标函数加速

本章节代码见于 [example_function_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py), [example_method_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_method_modes.py)


为了提升速度，**scikit-opt** 支持3种提升速度的方案：**矢量化**，**并行化**，**缓存化**  
- **矢量化**：要求目标函数本身支持矢量化运算（详见代码）。矢量化运算拥有极高的性能，通常比并行化运算要快。算法中，每代对应1次矢量化运算
- **多线程**：对目标函数没什么要求，通常比一般运算要快。如果目标函数是 IO 密集型，能达到更优的性能
- **多进程**：对目标函数没什么要求，通常比一般运算要快。如果目标函数是 CPU 密集型，能达到更优的性能
- **缓存化**：把每次计算的输入和输出缓存下来，下次调用时，如果已经缓存中已经存在，那么直接取出结果，而不再调用。缓存化特别适用于输入值有限的情况，例如纯整数规划、迭代到后期的TSP问题等。

总的来说，性能上，**矢量化** 远远大于 **多线程/多进程** 大于 **不加速**，如果是输入值得可能个数有限，**缓存化** 远大于其他方案。


下面比较 **不加速**、**矢量化**、**多线程**、**多进程** 的性能：

see [/examples/example_function_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py)

```python
import numpy as np
from sko.GA import GA
import time
import datetime
from sko.tools import set_run_mode


def generate_costly_function(task_type='io_costly'):
    # generate a high cost function to test all the modes
    # cost_type can be 'io_costly' or 'cpu_costly'
    if task_type == 'io_costly':
        def costly_function():
            time.sleep(0.1)
            return 1
    else:
        def costly_function():
            n = 10000
            step1 = [np.log(i + 1) for i in range(n)]
            step2 = [np.power(i, 1.1) for i in range(n)]
            step3 = sum(step1) + sum(step2)
            return step3

    return costly_function


for task_type in ('io_costly', 'cpu_costly'):
    costly_function = generate_costly_function(task_type=task_type)


    def obj_func(p):
        costly_function()
        x1, x2 = p
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


    for mode in ('common', 'multithreading', 'multiprocessing'):
        set_run_mode(obj_func, mode)
        ga = GA(func=obj_func, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
        start_time = datetime.datetime.now()
        best_x, best_y = ga.run()
        print('on {task_type} task,use {mode} mode, costs {time_costs}s'
              .format(task_type=task_type, mode=mode,
                      time_costs=(datetime.datetime.now() - start_time).total_seconds()))

    # to use the vectorization mode, the function itself should support the mode.
    mode = 'vectorization'


    def obj_func2(p):
        costly_function()
        x1, x2 = p[:, 0], p[:, 1]
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


    set_run_mode(obj_func2, mode)
    ga = GA(func=obj_func2, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
    start_time = datetime.datetime.now()
    best_x, best_y = ga.run()
    print('on {task_type} task,use {mode} mode, costs {time_costs}s'
          .format(task_type=task_type, mode=mode,
                  time_costs=(datetime.datetime.now() - start_time).total_seconds()))
```

output:
>>on io_costly task,use common mode, costs 5.116588s  
on io_costly task,use multithreading mode, costs 3.113499s  
on io_costly task,use multiprocessing mode, costs 3.119855s  
on io_costly task,use vectorization mode, costs 0.604762s  
on cpu_costly task,use common mode, costs 1.625032s  
on cpu_costly task,use multithreading mode, costs 1.60131s  
on cpu_costly task,use multiprocessing mode, costs 1.673792s  
on cpu_costly task,use vectorization mode, costs 0.192595s 


下面比较  **不加速** 和 **缓存化** 的性能
```python
def obj_func4_1(p):
    time.sleep(0.1)  # say that this function is very complicated and cost 0.1 seconds to run
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


def obj_func4_2(p):
    time.sleep(0.1)  # say that this function is very complicated and cost 0.1 seconds to run
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


set_run_mode(obj_func4_2, 'cached')
ga4_1 = GA(func=obj_func4_1, n_dim=2, size_pop=6, max_iter=10, lb=[-2, -2], ub=[2, 2], precision=1)
ga4_2 = GA(func=obj_func4_2, n_dim=2, size_pop=6, max_iter=10, lb=[-2, -2], ub=[2, 2], precision=1)

start_time = datetime.datetime.now()
best_x, best_y = ga4_1.run()
print('common mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())

start_time = datetime.datetime.now()
best_x, best_y = ga4_2.run()
print('cache mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())
print('cache mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())

```

output:
>on io_costly task,use common mode, costs 6.120317s  
on io_costly task,use cached mode, costs 1.106842s  
on cpu_costly task,use common mode, costs 1.914744s  
on cpu_costly task,use cached mode, costs 0.222713s  


## 算子优化加速

主要手段是 **矢量化** 和 **逻辑化**  


对于可以矢量化的算子，`scikit-opt` 都尽量做了矢量化，并且默认调用矢量化的算子，且 **无须用户额外操作**。  

另外，考虑到有些算子矢量化后，代码可读性下降，因此矢量化前的算子也会保留，为用户进阶学习提供方便。  


### 0/1 基因的mutation
做一个mask，是一个与 `Chrom` 大小一致的0/1矩阵，如果值为1，那么对应位置进行变异（0变1或1变0）  
自然想到用整除2的方式进行  

```python
def mutation(self):
    # mutation of 0/1 type chromosome
    mask = (np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut) * 1
    self.Chrom = (mask + self.Chrom) % 2
    return self.Chrom
```
如此就实现了一次性对整个种群所有基因变异的矢量化运算。用pycharm的profile功能试了一下，效果良好

再次改进。我还嫌求余数这一步速度慢，画一个真值表

|A|mask：是否变异|A变异后|
|--|--|--|
|1|0|1|
|0|0|0|
|1|1|0|
|0|1|1|

发现这就是一个 **异或**
```python
def mutation2(self):
    mask = (np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut)
    self.Chrom ^= mask
    return self.Chrom
```
测试发现运行速度又快了1~3倍，与最原始的双层循环相比，**快了约20倍**。  



### 0/1基因的crossover
同样思路，试试crossover.
- mask同样，1表示对应点交叉，0表示对应点不交叉


做一个真值表，总共8种可能，发现其中只有2种可能基因有变化（等位基因一样时，交叉后的结果与交叉前一样）

|A基因|B基因|是否交叉|交叉后的A基因|交叉后的B基因|
|--|--|--|--|--|
|1|0|1|0|1|
|0|1|1|1|0|

可以用 `异或` 和 `且` 来表示是否变化的表达式: `mask = (A^B)&C`，然后可以计算了`A^=mask, B^=mask`

代码实现
```
def crossover_2point_bit(self):
    Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
    Chrom1, Chrom2 = Chrom[::2], Chrom[1::2]
    mask = np.zeros(shape=(int(size_pop / 2),len_chrom),dtype=int)
    for i in range(int(size_pop / 2)):
        n1, n2 = np.random.randint(0, self.len_chrom, 2)
        if n1 > n2:
            n1, n2 = n2, n1
        mask[i, n1:n2] = 1
    mask2 = (Chrom1 ^ Chrom2) & mask
    Chrom1^=mask2
    Chrom2^=mask2
    Chrom[::2], Chrom[1::2]=Chrom1,Chrom2
    self.Chrom=Chrom
    return self.Chrom
```
测试结果，**效率提升约1倍**。


### 锦标赛选择算子selection_tournament
实战发现，selection_tournament 往往是最耗时的，几乎占用一半时间，因此需要优化。  
优化前的算法是遍历，每次选择一组进行锦标赛。但可以在二维array上一次性操作。
```python
def selection_tournament_faster(self, tourn_size=3):
    '''
    Select the best individual among *tournsize* randomly chosen
    Same with `selection_tournament` but much faster using numpy
    individuals,
    :param self:
    :param tourn_size:
    :return:
    '''
    aspirants_idx = np.random.randint(self.size_pop, size=(self.size_pop, tourn_size))
    aspirants_values = self.FitV[aspirants_idx]
    winner = aspirants_values.argmax(axis=1)  # winner index in every team
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
    self.Chrom = self.Chrom[sel_index, :]
    return self.Chrom
```

发现own time 和time 都降为原来的10%~15%，**效率提升了约9倍**
