Genetic Algorithm, PSO in Python



## Genetic Algorithm
```py
def demo_func2(p):
    x, y, z = p
    return -(x ** 2 + y ** 2 + z ** 2)
```

```py
general_best,func_general_best,FitV_history=ga.ga(func=demo_func2,  lb=[-1, -10, -5], ub=[2, 10, 2])
print('best_x:',general_best)
print('best_y:',func_general_best)

ga.plot_FitV(FitV_history)
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
