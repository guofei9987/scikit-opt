
## genetic algorithm do integer programming

If you want some variables to be integer, then set the corresponding `precision` to an integer
For example, our objective function is `demo_func`. We want the variables to be integer interval 2, integer interval 1, float. We set `precision=[2, 1, 1e-7]`:
```python
from sko.GA import GA

demo_func = lambda x: (x[0] - 1) ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=[-1, -1, -1], ub=[5, 1, 1], precision=[2, 1, 1e-7])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```

Notice:
- if `precision` is an integer, and the number of all possible value is $2^n$, the performance is best
- if `precision` is an integer, and the number of all possible value is not $2^n$, `GA` do these:
    1. Modify `ub` bigger, making the number of all possible value to be $2^n$
    2. Add an **unequal constraint**, and use penalty function to deal with it
    3. If your **equal constraint** `constraint_eq` 和 **unequal constraint** `constraint_ueq` is too much, the performance is not too good. you may want to manually deal with it.
- If `precision` is not an integer, but you still want this mode, manually deal with it. For example, your original `precision=0.5`, just make a new variable, multiplied by `2`
