## speed up objective function

### Vectorization calculation
If the objective function supports vectorization, it can run much faster.
The following `schaffer1` is an original objective function, `schaffer2` is the corresponding function that supports vectorization operations.  
`schaffer2.is_vector = True` is used to tell the algorithm that it supports vectorization operations, otherwise it is non-vectorized by default.  
As a result of the operation, the **time cost was reduced to 30%**  

```python
import numpy as np
import time


def schaffer1(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


def schaffer2(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p[:, 0], p[:, 1]
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


schaffer2.is_vector = True
# %%
from sko.GA import GA

ga1 = GA(func=schaffer1, n_dim=2, size_pop=5000, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
ga2 = GA(func=schaffer2, n_dim=2, size_pop=5000, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)

time_start = time.time()
best_x, best_y = ga1.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
print('time:', time.time() - time_start, ' seconds')

time_start = time.time()
best_x, best_y = ga2.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
print('time:', time.time() - time_start, ' seconds')
```
output:
>best_x: [-2.98023233e-08 -2.98023233e-08]  
 best_y: [1.77635684e-15]  
time: 88.32313132286072  seconds  


>best_x: [2.98023233e-08 2.98023233e-08]  
 best_y: [1.77635684e-15]  
time: 27.68204379081726  seconds  


`scikit-opt` still supports non-vectorization, because some functions are difficult to write as vectorization, and some functions are much less readable when vectorized.