Genetic Algorithm, PSO in Python



## Genetic Algorithm
```py
from ga import GA


def demo_func(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


ga = GA(func=demo_func, lb=[-1, -10, -5], ub=[2, 10, 2], max_iter=500)
best_x, best_y = ga.fit()
```

### plot the result using matplotlib
```py
import pandas as pd
import matplotlib.pyplot as plt
FitV_history = ga.FitV_history
FitV_history = pd.DataFrame(FitV_history)
fig, ax = plt.subplots(3, 1)
ax[0].plot(FitV_history.index, FitV_history.values, '.', color='red')
plt_mean = FitV_history.mean(axis=1)
plt_max = FitV_history.max(axis=1)
ax[1].plot(plt_mean.index, plt_mean, label='mean')
ax[1].plot(plt_max.index, plt_max, label='max')
ax[1].set_title('mean and all fitness of every generation')
ax[1].legend()

ax[2].plot(plt_max.index, plt_max.cummax())
ax[2].set_title('best fitness of every generation')
plt.show()
```

![Figure_1-1](https://i.imgur.com/yT7lm8a.png)


## PSO


```py
def demo_func2(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put
```

```py
from pso import PSO
my_pso = PSO(func=demo_func2, pop=30, dim=5, max_iter=100)
fitness = my_pso.fit()
print(my_pso.gbest_x)
print(my_pso.gbest_y)
my_pso.plot_history()
```
![Figure_1-1](https://i.imgur.com/4C9Yjv7.png)
