Genetic Algorithm, PSO in Python
## Define a Function
```py
def demo_func2(p):
    x, y, z = p
    return -(x ** 2 + y ** 2 + z ** 2)
```

## Genetic Algorithm
```py
general_best,func_general_best,FitV_history=ga.ga(func=demo_func2,  lb=[-1, -10, -5], ub=[2, 10, 2])
print('best_x:',general_best)
print('best_y:',func_general_best)

ga.plot_FitV(FitV_history)
```

![Figure_1-1](https://i.imgur.com/yT7lm8a.png)


## PSO
