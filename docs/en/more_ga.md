
## genetic algorithm do integer programming

If you want some variables to be 0/1 type, then you set the corresponding `precision` to an integer
For example, our objective function is `demo_func`. We want its first variable to be 0/1 type. We do as follows:
```python
from sko.GA import GA

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=[0, 0, 0], ub=[1, 1, 1], precision=[1, 1e-7, 1e-7])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```