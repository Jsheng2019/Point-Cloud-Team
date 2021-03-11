import numpy as np
from scipy.spatial import KDTree
from math import sin, cos
import matplotlib.pyplot as plt
import torchvision
import transforms
from torch.utils.data import DataLoader
from ModelNet40 import ModelNet40
from Perturbation import Perturbation
import math
import torch
from mpl_toolkits.mplot3d import Axes3D

# 构建KD树进行最近邻搜索
def RTtransform(point_source, point_target):
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
            R, t = RTtransform(S, targets)                #得到旋转矩阵和平移向量
            new_S = np.dot(R, S.T).T + t                #将新变换的点云代替之前的源点云
            if np.sum(np.abs(S - new_S)) < 1.0e-7:
                break
            S = np.copy(new_S)
        return S



if __name__ == '__main__':
    rootdir = r"E:\AI\Dataset\ModelNet40"
    perturbations = np.loadtxt(r"./model_perturbation/pert_6.csv", delimiter=',')     # 在model_perturbation选择变换文件（扭曲向量或者变换矩阵）
    transform = torchvision.transforms.Compose([
        transforms.Mesh2Points(),
        transforms.OnUnitCube(),
    ])
    testdata = ModelNet40(rootdir=rootdir, pattern='test/*.off', transform=transform)
    testdata = Perturbation(testdata, perturbation=perturbations, format_translate=False)
    testdataloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for i, data in enumerate(testdataloader):
        p0, p1, igt = data
        np_p0 = p0.cpu().contiguous().squeeze(0).numpy()  # --> (N, 3)
        np_p1 = p1.cpu().contiguous().squeeze(0).numpy()  # --> (M, 3)
        mod = ICP(np_p0, np_p1)
        g = mod.compute(2)     #迭代2次
        fig = plt.gcf()
        ax = Axes3D(fig)
        ax.set_axis_off()  # 清除坐标轴

        ax.scatter(g[:, 1], g[:, 0], g[:, 2], c='g', s=1)
        ax.scatter(np_p0[:, 1], np_p0[:, 0], np_p0[:, 2], c='r', s=1)
        ax.scatter(np_p1[:, 1], np_p1[:, 0], np_p1[:, 2], c='b', s=1)
        plt.show()
        break

