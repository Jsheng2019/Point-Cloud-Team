import numpy as np
root1 = '/media/ai/2.0TB_2/liujinsheng/PCR-MESH/ModelNet40_MeshNet/airplane/train/airplane_0450/face.npy'
root2 = '/media/ai/2.0TB_2/liujinsheng/PCR-MESH/ModelNet40_MeshNet/airplane/train/airplane_0450/neighbor_angle.npy'
data1 = np.load(root1)
data2 = np.load(root2)
#print(data[0])
print(len(data2), data2[0])