import numpy as np
import copy


"""
off文件的格式
第一行固定字符串OFF
第二行三个数字：顶点数，面片数，边数
第三行之后每一个顶点的坐标：x,y,z
顶点之后的所有行每一行列出一个面片：顶点个数，顶点标号（顶点坐标索引）
"""
FILEPATH = r"E:\AI\Dataset\ModelNet40\airplane\train\airplane_0001.off"

def getVertexsandMeshs(filepath):
    mesh=Mesh()
    with open(filepath, 'r') as file:
        sign = file.readline().strip()  # OFF
        if sign=="OFF":
            line = file.readline().strip()
            nums_vertices, nums_meshs, nums_edges = line.split()
            for v in range(int(nums_vertices)):
                ve=tuple(float(i) for i in file.readline().strip().split())
                mesh.vertices.append(ve)
            for m in range(int(nums_meshs)):
                me=tuple(int(i) for i in file.readline().strip().split()[1:])
                mesh.meshs.append(me)
        else:
            print("Error Format")
    return mesh


class Mesh:
    def __init__(self):
        self.vertices=[]    #所有顶点的坐标,元组形式储存
        self.meshs=[]   #所有面片坐标索引,元组形式储存
        self.edge=[]

    def clone(self):
        return copy.deepcopy(self)
    def vertex_array(self):
        return np.array(self.vertices)   #(N,3)

    # 将所有的面片索引转化为坐标
    @staticmethod
    def faces2polygons(faces, vertices):
        return list(map(lambda face: list(map(lambda vidx: vertices[vidx], face)), faces))










