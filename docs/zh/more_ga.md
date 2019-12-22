
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
- 当 `precision` 为整数时，会启用整数规划模式。
- 在整数规划模式下，如果某个变量的取值可能个数是 $2^n$，不会对性能有影响
- 在整数规划模式下，如果某个变量的取值可能个数不是 $2^n$，`GA` 会做这些事：
    1. 调整 `ub`，使得可能取值扩展成 $2^n$ 个
    2. 增加一个 **不等式约束** `constraint_ueq`，并使用罚函数法来处理
    3. 如果你的 **等式约束** `constraint_eq` 和 **不等式约束** `constraint_ueq` 已经很多了，更加推荐先手动做调整，以规避可能个数不是 $2^n$这种情况，毕竟太多的约束会影响性能。
- 如果 `precision` 不是整数（例如是0.5）,则不会进入整数规划模式，如果还想用这个模式，那么把对应自变量乘以2，这样 `precision` 就是整数了。
