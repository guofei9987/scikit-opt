import numpy as np
from scipy import spatial


def function_for_TSP(num_points, seed=None):
    if seed:
        np.random.seed(seed=seed)

    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    # print('distance_matrix is: \n', distance_matrix)

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    return num_points, points_coordinate, distance_matrix, cal_total_distance
