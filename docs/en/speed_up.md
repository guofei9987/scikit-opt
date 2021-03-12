## speed up objective function

Codes in this section see [example_function_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py), [example_method_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_method_modes.py)

### Vectorization calculation
If the objective function supports vectorization, it can run much faster.
The following `schaffer1` is an original objective function, `schaffer2` is the corresponding function that supports vectorization operations.  
`schaffer2.is_vector = True` is used to tell the algorithm that it supports vectorization operations, otherwise it is non-vectorized by default.  
As a result of the operation, the **time cost was reduced to 30%**  

To boost speed performance, **scikit-opt** supports 3 ways to speed up the objective function: **vectorization**, **parallel**, **cached**
- **Vectorization** requires that the objective function support vectorization. If so, the vectorization will gain extreme performance.
- **Parallel** requires nothing. It is usually faster then the common way
- **Cached** cache all the inputs and outputs. If the input in the next call is already in the cache, this method will fetch out the corresponding output from the cache, instead of calling the function once again. Cached gains extreme performance if the number of possible inputs is not big, such as integer programming or TSP

Totally speaking, **vectorization** is much faster then **parallel**, which is faster then **common**. If the number of input is not big, **cached** is much better then other ways.

To compare the speed of **common**, **vectorization**, **parallel**:


```python
import numpy as np
from sko.GA import GA
import time
import datetime
from sko.tools import set_run_mode


# %% mode 1: common

def obj_func1(p):
    time.sleep(0.1)  # say that this function is very complicated and cost 0.1 seconds to run
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


ga1 = GA(func=obj_func1, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)

start_time = datetime.datetime.now()
best_x, best_y = ga1.run()
print('common mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())


# %% mode 2: vectorization

def obj_func2(p):
    time.sleep(0.1)  # say that this function is very complicated and cost 1 seconds to run
    x1, x2 = p[:, 0], p[:, 1]
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


set_run_mode(obj_func2, 'vectorization')

ga2 = GA(func=obj_func2, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
start_time = datetime.datetime.now()
best_x, best_y = ga2.run()
print('vector mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())


# %% mode 3：parallel


def obj_func3(p):
    time.sleep(0.1)  # say that this function is very complicated and cost 0.1 seconds to run
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


set_run_mode(obj_func3, 'parallel')
ga3 = GA(func=obj_func3, n_dim=2, size_pop=6, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
start_time = datetime.datetime.now()
best_x, best_y = ga3.run()
print('parallel mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())


```

output:
>common mode, time costs:  5.284991
vector mode, time costs:  0.608695
parallel mode, time costs:  1.114424

To compare the speed of **common** and **cached**:

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
>common mode, time costs:  6.29733
cache mode, time costs:  0.308823


