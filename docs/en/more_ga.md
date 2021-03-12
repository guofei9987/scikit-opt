
## Do integer programming with genetic algorithm 

If you want some variables to be integer, then set the corresponding `precision` to an integer
For example, our objective function is `demo_func`. We want the variables to be integer interval 2, integer interval 1, float. We set `precision=[2, 1, 1e-7]`:
```python
from sko.GA import GA

demo_func = lambda x: (x[0] - 1) ** 2 + (x[1] - 0.05) ** 2 + x[2] ** 2
ga = GA(func=demo_func, n_dim=3, max_iter=500, lb=[-1, -1, -1], ub=[5, 1, 1], precision=[2, 1, 1e-7])
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```

Notice:
- If `precision` is an integer, the number of all possible value would better be $2^n$, in which case the performance is the best. It also works if the number is not $2^n$
<!-- - if `precision` is an integer, and the number of all possible value is not $2^n$, `GA` do these:
    1. Modify `ub` bigger, making the number of all possible value to be $2^n$
    2. Add an **unequal constraint**, and use penalty function to deal with it
    3. If your **equal constraint** `constraint_eq` 和 **unequal constraint** `constraint_ueq` is too much, the performance is not too good. you may want to manually deal with it. -->
- If `precision` is not an integer, but you still want this mode, manually deal with it. For example, your original `precision=0.5`, just make a new variable, multiplied by `2`



## How to fix start point and end point with GA for TSP
If it is not a cycle graph, no need to do this.

if your start point and end point is (0, 0) and (1, 1). Build up the object function :
- Start point and end point is not the input of the object function. If totally n+2 points including start and end points, the input is the n points.
- And build up the object function, which is the total distance, as actually they are.


```python
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

num_points = 20

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
start_point=[[0,0]]
end_point=[[1,1]]
points_coordinate=np.concatenate([points_coordinate,start_point,end_point])
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    routine = np.concatenate([[num_points], routine, [num_points+1]])
    return sum([distance_matrix[routine[i], routine[i + 1]] for i in range(num_points+2-1)])
```

And the same with others:
```python
from sko.GA import GA_TSP

ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
best_points, best_distance = ga_tsp.run()


fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([[num_points],best_points, [num_points+1]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()
```

![image](https://user-images.githubusercontent.com/19920283/83831463-0ac6a400-a71a-11ea-8692-beac5f465111.png)

For more information, click [here](https://github.com/guofei9987/scikit-opt/issues/58)

## How to set up starting point or initial population

- For `GA`, after `ga=GA(**params)`, use codes like `ga.Chrom = np.random.randint(0,2,size=(80,20))` to manually set the initial population.  
- For `DE`, set `de.X` to your initial X.  
- For `SA`, there is a parameter `x0`, which is the init point.
- For `PSO`, set `pso.X` to your initial X, and run `pso.cal_y(); pso.update_gbest(); pso.update_pbest()`
