import torch

# Mesh->Points（Mesh的顶点编号转换为坐标）
class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh=mesh.clone()
        vertex_array=mesh.vertex_array() #(N,3)  N是点云中点的个数
        return torch.from_numpy(vertex_array).type(dtype=torch.float)


# 转换为单位立方体
class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True)  # (N,D)->(1,D)
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0]  # (N.D)->(D)
        s = torch.max(c)
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        return self.method2(tensor)
