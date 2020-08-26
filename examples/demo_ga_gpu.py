import numpy as np
import torch
import time


def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


import torch
from sko.GA import GA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
ga.to(device=device)
start_time = time.time()
best_x, best_y = ga.run()
print(time.time() - start_time)
print('best_x:', best_x, '\n', 'best_y:', best_y)

ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
start_time = time.time()
best_x, best_y = ga.run()
print(time.time() - start_time)
print('best_x:', best_x, '\n', 'best_y:', best_y)
