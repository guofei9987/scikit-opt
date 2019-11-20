
## 遗传算法进行0-1规划

在多维优化时，想让哪个变量限制为整数，就设定 `precision` 为 1即可。  
例如，我想让我的自定义函数 `demo_func` 的第一个变量限制为整数，那么就设定 `precision` 的第一个数为1，例子如下：

```python
from sko.GA import GA

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=[0, 0, 0], ub=[1, 1, 1], precision=[1, 1e-7, 1e-7])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```