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


# %%cache mode

def obj_func_for_cache_mode(p):
    costly_function()
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


for task_type in ('io_costly', 'cpu_costly'):
    costly_function = generate_costly_function(task_type=task_type)

    for mode in ('common', 'cached'):
        set_run_mode(obj_func_for_cache_mode, mode)
        ga_2 = GA(func=obj_func_for_cache_mode, n_dim=2, size_pop=6, max_iter=10, lb=[-2, -2], ub=[2, 2], precision=1)
        start_time = datetime.datetime.now()
        best_x, best_y = ga_2.run()
        print('on {task_type} task,use {mode} mode, costs {time_costs}s'
              .format(task_type=task_type, mode=mode,
                      time_costs=(datetime.datetime.now() - start_time).total_seconds()))
