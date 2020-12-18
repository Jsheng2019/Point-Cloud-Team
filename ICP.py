import numpy as np
from scipy.spatial import KDTree


# 构建KD树进行最近邻搜索
def transform(point_source, point_target):
    S = np.copy(point_source)  # (N,3)
    T = np.copy(point_target)  # (N,3)
    # 两个点云的质心
    centroid_S = np.mean(S, axis=0)
    centroid_T = np.mean(T, axis=0)

    S -= centroid_S
    T -= centroid_T

    H = np.dot(S.T, T)  # 需要SVD分解的矩阵
    U, Sigma, Vt = np.linalg.svd(H)
    V = Vt.T
    R = np.dot(V, U.T)  # 旋转矩阵

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_S) + centroid_T
    return R, t

class ICP:
    def __init__(self, pt, ps):
        self.pt = pt  # 模板点云
        self.ps = ps  # 源点云
        self.leafsize = 1000
        self.nearest = KDTree(self.pt, leafsize=self.leafsize)

    def compute(self, max_iteration):
        S = np.copy(self.ps)      #源点云作为查询点
        for iters in range(max_iteration):
            neighbor_idx = self.nearest.query(S)[1]     #在模板点云中离查询点（源点云）最近的点的源点云索引
            targets = self.pt[neighbor_idx]             #在模板点云对应的索引
            R, t = transform(S, targets)                #得到旋转矩阵和平移向量
            new_S = np.dot(R, S.T).T + t                #将新变换的点云代替之前的源点云
            if np.sum(np.abs(S - new_S)) < 1.0e-7:
                break
            S = np.copy(new_S)
        return S


def icp_test():
    from math import sin, cos
    import matplotlib.pyplot as plt

    Y, X = np.mgrid[0:100:5, 0:100:5]  # 生成二维结构，列表的第一个参数的按列展开，第二个参数按行展开
    Z = Y ** 2 + X ** 2
    A = np.vstack([Y.reshape(-1), X.reshape(-1), Z.reshape(-1)]).T

    R = np.array([
        [cos(-0.279), -sin(-0.279), 0],
        [sin(-0.279), cos(-0.279), 0],
        [0, 0, 1]
    ])

    t = np.array([5.0, 20.0, 10.0])

    B = np.dot(R, A.T).T + t
    A = A.astype(B.dtype)
    icp = ICP(A, B)

    points = icp.compute(10)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_label("x - axis")
    ax.set_label("y - axis")
    ax.set_label("z - axis")

    ax.plot(A[:, 1], A[:, 0], A[:, 2], "o", color="#cccccc", ms=4, mew=0.5)
    ax.plot(points[:, 1], points[:, 0], points[:, 2], "o", color="#00cccc", ms=4, mew=0.5)
    ax.plot(B[:, 0], B[:, 1], B[:, 2], "o", color="#ff0000", ms=4, mew=0.5)

    plt.show()


if __name__ == '__main__':
    icp_test()
