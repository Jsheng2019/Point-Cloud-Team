import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torch.utils.data import DataLoader
from operations.transform_functions import PCRNetTransform
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, os.path.pardir,'ModelNet40_MeshNet')


# def load_data(train):
#     if train:
#         partition = 'train'
#     else:
#         partition = 'test'
#
#     # 存放数据和对应标签
#     Data = []
#     Label = []
#     for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_{}*.h5'.format(partition))):
#         with h5py.File(h5_name, 'r') as file:
#             data = file['data'][:].astype('float32')
#             label = file['label'][:].astype('int64')
#             Data.append(data)
#             Label.append(label)
#
#     Data = np.concatenate(Data, axis=0)  # (9840, 2048, 3)  9840个样本，每个样本2048个点，每个点3维
#     Label = np.concatenate(Label, axis=0)  # (9840, 1)
#     return Data, Label
#
#
# def read_classed():
#     # 读取所有类的类名
#     with open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r') as file:
#         shape_name = file.read()
#         shape_name = np.array(shape_name.split('\n')[:-1])
#         return shape_name
#
#
# class ModelNet40(Dataset):
#     def __init__(self, train=True, num_points=1024, randomize_data=False):
#         super(ModelNet40, self).__init__()
#         self.data, self.labels = load_data(train)
#         self.shapes = read_classed()
#         self.num_points = num_points
#         self.randomize_data = randomize_data
#
#     def __getitem__(self, index):
#         if self.randomize_data:
#             current_points = self.randomize(index)  # 从该实例2048个点随机采样了1024个点
#         else:
#             current_points = self.data[index].copy()  # 直接使用该实例2048个点
#
#         current_points = torch.from_numpy(current_points).float()
#         label = torch.from_numpy(self.labels[index]).type(torch.LongTensor)
#         return current_points, label  # 返回该实例（实例从2048个点随机采样了1024个点）以及标签
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def get_shape(self, label):
#         return self.shapes[label]
#
#     def randomize(self, index):
#         point_index = np.arange(0, self.num_points)  # 在0~num_points范围内生成索引
#         np.random.shuffle(point_index)  # 打乱索引
#         return self.data[index, point_index].copy()
type_to_index_map = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39
}

class ModelNet40(Dataset):

    def __init__(self, part='train'):
        self.root = DATA_DIR
        self.augment_data = True
        self.max_faces = 1024
        self.part = part

        self.data = []
        for type in os.listdir(self.root):
            type_index = type_to_index_map[type]
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    self.data.append((os.path.join(type_root, filename), type_index))

    def __getitem__(self, i):
        path, type = self.data[i]
        data = np.load(path)
        face = data['face']
        neighbor_index = data['neighbor_index']

        # data1 augmentation
        if self.augment_data and self.part == 'train':
            sigma, clip = 0.01, 0.05
            jittered_data = np.clip(sigma * np.random.randn(*face[:, :12].shape), -1 * clip, clip)
            face = np.concatenate((face[:, :12] + jittered_data, face[:, 12:]), 1)

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        # target = torch.tensor(type, dtype=torch.long)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index

    def __len__(self):
        return len(self.data)

# data = ModelNet40()
# centers, corners, normals, neighbor_index = data[0]
# corner1 = corners[6:9]
# print(corners)


class RegistrationData(Dataset):
    def __init__(self, algorithm='iPCRNet', data_class=ModelNet40(), is_testing=False):
        super(RegistrationData, self).__init__()
        self.algorithm = algorithm
        self.is_testing = is_testing
        self.data_class = data_class
        if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
            self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)

    def __getitem__(self, index):
        # template_pc:模版点云 source_pc:源点云
        # template_pc, label = self.data_class[index]
        centers, corners, normals, neighbor_index = self.data_class[index]

        # 将三个角点向量分离(9*1024 >>> 3*1024 3*1024 3*1024)
        corner1 = corners[:3]
        corner2 = corners[3:6]
        corner3 = corners[6:9]



        # 交换维度顺序(3*1024 >>> 1024*3)

        centers_exc = centers.permute(1, 0).contiguous()
        corner1_exc = corner1.permute(1, 0).contiguous()
        corner2_exc = corner2.permute(1, 0).contiguous()
        corner3_exc = corner3.permute(1, 0).contiguous()
        normals_exc = normals.permute(1, 0).contiguous()
        self.transforms.index = index

        corners_cat = torch.cat((corner1_exc, corner2_exc, corner3_exc), 0)

        # 调用__call__对模版点云变换后获得源点云
        # source_pc = self.transforms(template_pc)
        # 对分离出的中心点、角点、法向量施加变换
        centers_trans = self.transforms(centers_exc)
        corner1_trans = self.transforms(corner1_exc)
        corner2_trans = self.transforms(corner2_exc)
        corner3_trans = self.transforms(corner3_exc)
        normals_trans = self.transforms(normals_exc)
        corners_cat_trans = torch.cat((corner1_trans, corner2_trans, corner3_trans), 0)
        # corners_cat_trans = np.array(list(set([tuple(t) for t in corners_cat_trans])))
        #print(corners_cat_trans.shape)
        # corners_cat_trans = torch.tensor(torch.from_numpy(corners_cat_trans), dtype=torch.float32)

        #按原来的组合形式合并变换后的中心点、角点、法向量
        #tensordata = torch.from_numpy(numpydata)

        centers_trans = centers_trans.permute(1,0).contiguous()
        corners_trans = torch.cat((corner1_trans, corner2_trans, corner3_trans), 1).permute(1,0).contiguous()
        normals_trans = normals_trans.permute(1,0).contiguous()

        template_mesh = [centers, corners, normals, neighbor_index, corners_cat]
        source_mesh = [centers_trans, corners_trans, normals_trans, neighbor_index, corners_cat_trans]

        igt = self.transforms.igt
        if self.is_testing:
            #返回列表：模板点云，源点云，真实的变换矩阵7d，真实的旋转矩阵，真实的平移向量
            return template_mesh, source_mesh, igt, self.transforms.igt_rotation, self.transforms.igt_translation
        else:
            # 返回列表：模板点云，源点云，真实的变换矩阵7d
            return template_mesh, source_mesh, igt, self.transforms.igt_rotation, self.transforms.igt_translation

    def __len__(self):
        return len(self.data_class)


if __name__ == '__main__':
    data = RegistrationData('PCRNet', ModelNet40(train=False))
    test_loader = DataLoader(data, batch_size=1, shuffle=False)
    for i, data in enumerate(test_loader):
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        break
