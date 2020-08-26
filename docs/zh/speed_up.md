## 目标函数加速

### 矢量化计算
如果目标函数支持矢量化运算，那么运行速度可以大大加快。  
下面的 `schaffer1` 是普通的目标函数，`schaffer2` 是支持矢量化运算的目标函数，需要用`schaffer2.is_vector = True`来告诉算法它支持矢量化运算，否则默认是非矢量化的。  
从运行结果看，花费时间降低到30%
```python
import numpy as np
import time


def schaffer1(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


def schaffer2(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p[:, 0], p[:, 1]
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


schaffer2.is_vector = True
# %%
from sko.GA import GA

ga1 = GA(func=schaffer1, n_dim=2, size_pop=5000, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
ga2 = GA(func=schaffer2, n_dim=2, size_pop=5000, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)

time_start = time.time()
best_x, best_y = ga1.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
print('time:', time.time() - time_start, ' seconds')

time_start = time.time()
best_x, best_y = ga2.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
print('time:', time.time() - time_start, ' seconds')
```
output:
>best_x: [-2.98023233e-08 -2.98023233e-08]  
 best_y: [1.77635684e-15]  
time: 88.32313132286072  seconds  


>best_x: [2.98023233e-08 2.98023233e-08]  
 best_y: [1.77635684e-15]  
time: 27.68204379081726  seconds  


`scikit-opt` 仍然支持非矢量化计算，因为有些函数很难写成矢量化计算的形式，还有些函数强行写成矢量化形式后可读性会大大降低。

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
