
## demonstrate PSO with animation

step1:do pso  
-> Demo code: [examples/demo_pso_ani.py#s1](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso_ani.py#L1)
```python
# Plot particle history as animation
import numpy as np
from sko.PSO import PSO


def demo_func(x):
    x1, x2 = x
    return x1 ** 2 + (x2 - 0.05) ** 2


pso = PSO(func=demo_func, dim=2, pop=20, max_iter=40, lb=[-1, -1], ub=[1, 1])
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

```

step2: plot animation  
-> Demo code: [examples/demo_pso_ani.py#s2](https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_pso_ani.py#L1)
```python
# Plot particle history as animation
import numpy as np
from sko.PSO import PSO


def demo_func(x):
    x1, x2 = x
    return x1 ** 2 + (x2 - 0.05) ** 2


pso = PSO(func=demo_func, dim=2, pop=20, max_iter=40, lb=[-1, -1], ub=[1, 1])
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

```

![pso_ani](https://github.com/guofei9987/pictures_for_blog/blob/master/heuristic_algorithm/pso.gif?raw=true)  