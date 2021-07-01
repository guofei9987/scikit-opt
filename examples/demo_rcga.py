import numpy as np

def schaffer(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


from sko.GA import RCGA

rcga = RCGA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, prob_cros=0.9, prob_mut=0.1, lb=[-1, -1], ub=[1, 1])
best_x, best_y = rcga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(rcga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
