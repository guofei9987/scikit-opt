import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6)
num_points = 8

points = range(num_points)
points_coordinate = np.random.rand(num_points, 2)
distance_matrix = np.zeros(shape=(num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        distance_matrix[i][j] = np.linalg.norm(points_coordinate[i] - points_coordinate[j], ord=2)
print('distance_matrix is: \n', distance_matrix)


def demo_func(points):
    num_points, = points.shape
    total_distance = 0
    for i in range(num_points - 1):
        total_distance += distance_matrix[points[i], points[i + 1]]
    total_distance += distance_matrix[points[i + 1], points[0]]
    return total_distance


# %%
import numpy as np

func = len(points)
n = 8  # 城市数量
m = 20  # 蚂蚁数量
alpha = 1  # 信息素重要程度
beta = 1  # 适应度的重要程度
rho = 0.1  # 信息素挥发速度

iter_max = 800
Tau = np.ones((n, n))  # 信息素矩阵
Table = np.zeros((m, n)).astype(np.int)  # 一代每个蚂蚁的实际路径

# x_g_best, y_best = [], []  # 记录各代的最佳情况
x_best_history, y_best_history = [], []

# %%
for i in range(n):
    distance_matrix[i, i] = 1e-10  # 避免除零错误

for i in range(iter_max):  # 对每次迭代
    prob_matrix = (Tau ** alpha) * (1 / distance_matrix) ** beta  # 转移概率，无须归一化。
    for j in range(m):  # 对每个蚂蚁
        Table[j, 0] = 0  # start point，其实可以随机，但没什么区别
        for k in range(n - 1):  # 蚂蚁到达的每个节点
            taboo_set = set(Table[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
            allow_list = list(set(points) - taboo_set)  # 在这些点中做选择
            prob = prob_matrix[Table[j, k], allow_list]
            prob = prob / prob.sum()
            next_point = np.random.choice(allow_list, size=1, p=prob)[0]
            Table[j, k + 1] = next_point

    # 计算距离
    y = np.array([demo_func(i) for i in Table])

    # 顺便记录历史最好情况
    index_best = y.argmin()
    x_best, y_best = Table[index_best, :], y[index_best]
    x_best_history.append(x_best)
    y_best_history.append(y_best)

    # 计算需要新涂抹的信息素
    delta_tau = np.zeros((n, n))
    for j in range(m):  # 每个蚂蚁
        for k in range(n - 1):  # 每个节点
            n1, n2 = Table[j, k], Table[j, k + 1]
            delta_tau[n1, n2] += 1 / y[j]
        n1, n2 = Table[j, n - 1], Table[j, 0]
        delta_tau[n1, n2] += 1 / y[j]

    # 信息素飘散+信息素涂抹
    Tau = (1 - rho) * Tau + delta_tau

Tau
# %%
y_best_history = np.array(y_best_history)
a = y_best_history.argmin()
best_points = x_best_history[a]

# %%
fig, ax = plt.subplots(1, 1)
plt.plot(pd.DataFrame(y_best_history).cummin(axis=0))

fig, ax = plt.subplots(1, 1)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
plt.show()
