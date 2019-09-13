# demo: GA for TSP with UDF(user defined functions)
import numpy as np
# step1: randomly generate the data
from sko.demo_func import function_for_TSP

num_points, points_coordinate, distance_matrix, cal_total_distance = function_for_TSP(num_points=15)

# step2: DO GA with UDF
from sko.GA import GA_TSP, ga_with_udf
from sko.GA import selection_tournament, mutation_TSP_1

options = {'selection': {'udf': selection_tournament, 'kwargs': {'tourn_size': 3}},
           'mutation': {'udf': mutation_TSP_1}}
GA_TSP_demo = ga_with_udf(GA_TSP, options)

ga_tsp = GA_TSP_demo(func=cal_total_distance, n_dim=num_points, pop=500, max_iter=800, Pm=0.2)

best_points, best_distance = ga_tsp.fit()

# step3: plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()


