import numpy as np
import time
import datetime
from sko.GA import GA
from sko.tools import set_run_mode


class YourClass:
    def __init__(self, param=1):
        self.param = param

    def obj_func1(self, p):
        time.sleep(0.1)  # say that this function is very complicated and cost 0.1 seconds to run
        x1, x2 = p
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)

    def obj_func2(self, p):
        time.sleep(0.1)  # say that this function is very complicated and cost 1 seconds to run
        x1, x2 = p[:, 0], p[:, 1]
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)

    def obj_func3(self, p):
        x1, x2 = p
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(self.param + 0.001 * x)

    def obj_func4_1(self, p):
        time.sleep(0.1)  # say that this function is very complicated and cost 0.1 seconds to run
        x1, x2 = p
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)

    def obj_func4_2(self, p):
        time.sleep(0.1)  # say that this function is very complicated and cost 0.1 seconds to run
        x1, x2 = p
        x = np.square(x1) + np.square(x2)
        return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


your_class = YourClass()

set_run_mode(your_class.obj_func2, 'vectorization')
set_run_mode(your_class.obj_func3, 'multithreading')

ga1 = GA(func=your_class.obj_func1, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
ga2 = GA(func=your_class.obj_func2, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)
ga3 = GA(func=your_class.obj_func3, n_dim=2, size_pop=10, max_iter=5, lb=[-1, -1], ub=[1, 1], precision=1e-7)

start_time = datetime.datetime.now()
best_x, best_y = ga1.run()
print('common mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())

start_time = datetime.datetime.now()
best_x, best_y = ga2.run()
print('vector mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())

start_time = datetime.datetime.now()
best_x, best_y = ga3.run()
print('parallel mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())

# %%

set_run_mode(your_class.obj_func4_2, 'cached')
ga4_1 = GA(func=your_class.obj_func4_1, n_dim=2, size_pop=6, max_iter=10, lb=[-2, -2], ub=[2, 2], precision=1)
ga4_2 = GA(func=your_class.obj_func4_2, n_dim=2, size_pop=6, max_iter=10, lb=[-2, -2], ub=[2, 2], precision=1)

start_time = datetime.datetime.now()
best_x, best_y = ga4_1.run()
print('common mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())

start_time = datetime.datetime.now()
best_x, best_y = ga4_2.run()
print('cache mode, time costs: ', (datetime.datetime.now() - start_time).total_seconds())
