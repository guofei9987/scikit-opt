## speed up objective function

Codes in this section see [example_function_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py), [example_method_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_method_modes.py)


To boost speed performance, **scikit-opt** supports 3 ways to speed up the objective function: **vectorization**, **parallel**, **cached**
- **Vectorization** requires that the objective function support vectorization. If so, the vectorization will gain extreme performance.
- **multithreading** requires nothing. It is usually faster than the common way, better than multiprocessing in io-intensive function
- **multiprocessing** requires nothing. It is usually faster than the common way, better than multithreading in io-intensive function
- **Cached** cache all the inputs and outputs. If the input in the next call is already in the cache, this method will fetch out the corresponding output from the cache, instead of calling the function once again. Cached gains extreme performance if the number of possible inputs is not big, such as integer programming or TSP

Totally speaking, **vectorization** is much faster then **parallel**, which is faster then **common**. If the number of input is not big, **cached** is much better then other ways.

To compare the speed of **common**, **vectorization**, **parallel**:


```python
import numpy as np
from sko.GA import GA
import time
import datetime
from sko.tools import set_run_mode


def generate_costly_function(task_type='io_costly'):
    # generate a high cost function to test all the modes
    # cost_type can be 'io_costly' or 'cpu_costly'
    if task_type == 'io_costly':
        def costly_function():
            time.sleep(0.1)
            return 1
    else:
        def costly_function():
            n = 10000
            step1 = [np.log(i + 1) for i in range(n)]
            step2 = [np.power(i, 1.1) for i in range(n)]
            step3 = sum(step1) + sum(step2)
            return step3

    return costly_function


for task_type in ('io_costly', 'cpu_costly'):
    costly_function = generate_costly_function(task_type=task_type)


    def obj_func(p):
        costly_function()
        x1, x2 = p
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


    for mode in ('common', 'multithreading', 'multiprocessing'):
        set_run_mode(obj_func, mode)
        ga = GA(func=obj_func, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
        start_time = datetime.datetime.now()
        best_x, best_y = ga.run()
        print('on {task_type} task,use {mode} mode, costs {time_costs}s'
              .format(task_type=task_type, mode=mode,
                      time_costs=(datetime.datetime.now() - start_time).total_seconds()))

    # to use the vectorization mode, the function itself should support the mode.
    mode = 'vectorization'


    def obj_func2(p):
        costly_function()
        x1, x2 = p[:, 0], p[:, 1]
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


    set_run_mode(obj_func2, mode)
    ga = GA(func=obj_func2, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
    start_time = datetime.datetime.now()
    best_x, best_y = ga.run()
    print('on {task_type} task,use {mode} mode, costs {time_costs}s'
          .format(task_type=task_type, mode=mode,
                  time_costs=(datetime.datetime.now() - start_time).total_seconds()))

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


