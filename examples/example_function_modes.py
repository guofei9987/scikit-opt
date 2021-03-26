import numpy as np
from sko.GA import GA
import time
import datetime
from sko.tools import set_run_mode




def generate_costly_function(cost_type='io_costly'):
    # generate a high cost function to test all the modes
    # cost_type can be 'io_costly' or 'cpu_costly'
    if cost_type == 'io_costly':
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


costly_function = generate_costly_function(cost_type='io_costly')


# %% type1: common

def obj_func1(p):
    costly_function()
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


ga1 = GA(func=obj_func1, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)

start_time = datetime.datetime.now()
best_x, best_y = ga1.run()
print('common mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())


# %% type2:矢量化

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


# %% type3：parallel


def obj_func3(p):
    costly_function()
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)

set_run_mode(obj_func3, 'parallel')
ga3 = GA(func=obj_func3, n_dim=2, size_pop=6, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
start_time = datetime.datetime.now()
best_x, best_y = ga3.run()
print('parallel mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())

# %% type3_2: multiprocess (get error on windows)


def obj_func3_2(p):
    costly_function()
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


set_run_mode(obj_func3_2, 'multiprocess')
ga3_2 = GA(func=obj_func3_2, n_dim=2, size_pop=6, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
start_time = datetime.datetime.now()
best_x, best_y = ga3_2.run()
print('multiprocess mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())


# %%cache mode


def obj_func4_1(p):
    costly_function()
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


def obj_func4_2(p):
    costly_function()
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
