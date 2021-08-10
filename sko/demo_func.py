import numpy as np
from scipy import spatial


def function_for_TSP(num_points, seed=None):
    if seed:
        np.random.seed(seed=seed)

    # generate coordinate of points randomly
    points_coordinate = np.random.rand(num_points, 2)
    distance_matrix = spatial.distance.cdist(
        points_coordinate, points_coordinate, metric='euclidean')

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
    二维函数，具有无数个极小值点、强烈的震荡形态。很难找到全局最优值
    在(0,0)处取的最值0
    -10<=x1,x2<=10
    '''
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(np.sqrt(x))) - 0.5) / np.square(1 + 0.001 * x)


def shubert(p):
    '''
    2-dimension
    -10<=x1,x2<=10
    has 760 local minimas, 18 of which are global minimas with -186.7309
    '''
    x, y = p
    part1 = [i * np.cos((i + 1) * x + i) for i in range(1, 6)]
    part2 = [i * np.cos((i + 1) * y + i) for i in range(1, 6)]
    return np.sum(part1) * np.sum(part2)


def griewank(p):
    '''
    存在多个局部最小值点，数目与问题的维度有关。
    此函数是典型的非线性多模态函数，具有广泛的搜索空间，是优化算法很难处理的复杂多模态问题。
    在(0,...,0)处取的全局最小值0
    -600<=xi<=600
    '''
    part1 = [np.square(x) / 4000 for x in p]
    part2 = [np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(p)]
    return np.sum(part1) - np.prod(part2) + 1


def rastrigrin(p):
    '''
    多峰值函数，也是典型的非线性多模态函数
    -5.12<=xi<=5.12
    在范围内有10n个局部最小值，峰形高低起伏不定跳跃。很难找到全局最优
    has a global minimum at x = 0  where f(x) = 0
    '''
    return np.sum([np.square(x) - 10 * np.cos(2 * np.pi * x) + 10 for x in p])


def rosenbrock(p):
    '''
    -2.048<=xi<=2.048
    函数全局最优点在一个平滑、狭长的抛物线山谷内，使算法很难辨别搜索方向，查找最优也变得十分困难
    在(1,...,1)处可以找到极小值0
    :param p:
    :return:
    '''
    n_dim = len(p)
    res = 0
    for i in range(n_dim - 1):
        res += 100 * \
            np.square(np.square(p[i]) - p[i + 1]) + np.square(p[i] - 1)
    return res

def sixhumpcamel(p):
    """
    带域的 2dim 的多模态全局最小化函数
    -5<=xi<=5,
    f(-0.08..., 0.712...) 的全局最小值为 -1.0...4
    """
    x,y=p
    return 4*np.square(x)+ x*y -4*np.square(y) -2.1*np.power(x,4) + 4*np.power(y,4) +1/3*np.power(x,6)

def zakharov(p):
    """
    它是一个具有范围的 n 维单峰函数
    -5<=xi<=10
    除了全局最小值之外，该函数没有局部最小值。It
    The global minimum can be found at 0, for f(0, ..., 0).
    :param p:
    :return:
    """
    temp2 = [0.5*i*x for i, x in enumerate(p)]
    part2 = np.sum(temp2)

    temp1 = [np.square(x) for x in p]
    part1 = np.sum(temp1)
    return part1 + part2**2 + part2**4


def ackley(p):
    """ Ackley_N.2
    -32<=xi<=32. Convex 2dim , non-seperable function .
    The global minimum value -200 can be found at f(0,0)
    :param p:
    :return:
    """
    x, y = p
    return -200 * np.exp(-0.02 * np.sqrt(np.square(x)) + np.square(y))

def cigar(p):
    """  
    多峰全局优化函数，域为-100<=xi<=100，对于i=1...n。
    f(0,...0) 的全局最小值为 0
    """
    x=p
    return np.square(float(x[0])) + np.power(10.0,6) * sphere(x[1:])

if __name__ == '__main__':
    print(sphere((0, 0)))
    print(schaffer((0, 0)))
    print(shubert((-7.08350643, -7.70831395)))
    print(griewank((0, 0, 0)))
    print(rastrigrin((0, 0, 0)))
    print(rosenbrock((1, 1, 1)))
    print(zakharov((0, 0, 0)))
    print(ackley((0, 0)))
    print(cigar((0,0,0,0,)))
    print(sixhumpcamel((-0.08984201368301331, 0.7126564032704135)))
