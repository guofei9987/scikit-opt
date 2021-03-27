## speed up objective function

Codes in this section see [example_function_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py), [example_method_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_method_modes.py)


To boost speed performance, **scikit-opt** supports 3 ways to speed up the objective function: **vectorization**, **parallel**, **cached**
- **Vectorization** requires that the objective function support vectorization. If so, the vectorization will gain extreme performance.
- **multithreading** requires nothing. It is usually faster than the common way, better than multiprocessing in io-intensive function
- **multiprocessing** requires nothing. It is usually faster than the common way, better than multithreading in io-intensive function
- **Cached** cache all the inputs and outputs. If the input in the next call is already in the cache, this method will fetch out the corresponding output from the cache, instead of calling the function once again. Cached gains extreme performance if the number of possible inputs is not big, such as integer programming or TSP

Totally speaking, **vectorization** is much faster then **parallel**, which is faster then **common**. If the number of input is not big, **cached** is much better then other ways.

To compare the speed of **common**, **vectorization**, **parallel**:

see [/examples/example_function_modes.py](https://github.com/guofei9987/scikit-opt/blob/master/examples/example_function_modes.py)

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
>on io_costly task,use common mode, costs 5.116588s  
on io_costly task,use multithreading mode, costs 3.113499s  
on io_costly task,use multiprocessing mode, costs 3.119855s  
on io_costly task,use vectorization mode, costs 0.604762s  
on cpu_costly task,use common mode, costs 1.625032s  
on cpu_costly task,use multithreading mode, costs 1.60131s  
on cpu_costly task,use multiprocessing mode, costs 1.673792s  
on cpu_costly task,use vectorization mode, costs 0.192595s  


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
>on io_costly task,use common mode, costs 6.120317s  
on io_costly task,use cached mode, costs 1.106842s  
on cpu_costly task,use common mode, costs 1.914744s  
on cpu_costly task,use cached mode, costs 0.222713s  


