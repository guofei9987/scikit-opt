import numpy as np
from scipy import spatial


def function_for_TSP(num_points, seed=None):
    if seed:
        np.random.seed(seed=seed)

    points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points randomly
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    # print('distance_matrix is: \n', distance_matrix)

    def cal_total_distance(routine):
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

    return num_points, points_coordinate, distance_matrix, cal_total_distance


def sphere(p):
    # Sphere函数
    out_put = 0
    for i in p:
        out_put += i ** 2
    return out_put


def schaffer(p):
    '''
    此函数具有无数个极小值点、强烈的震荡形态。很难找到全局最优值
    在(0,0)处取的最值0
    :param p:
    :return:
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)
