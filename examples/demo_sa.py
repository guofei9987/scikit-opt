from sko.SA import SA

demo_func = lambda x: x[0] ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
sa = SA(func=demo_func, x0=[1, 1, 1])
x_star, y_star = sa.fit()
print(x_star, y_star)

import matplotlib.pyplot as plt
import pandas as pd

plt.plot(pd.DataFrame(sa.f_list).cummin(axis=0))
plt.show()
