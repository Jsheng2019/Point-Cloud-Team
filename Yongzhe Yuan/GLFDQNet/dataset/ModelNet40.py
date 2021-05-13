import torch
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import glob
from torch.utils.data import DataLoader
from operations.transform_functions import PCRNetTransform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, os.pardir,'data')


def load_data(train):
    if train:
        partition = 'train'
    else:
        partition = 'test'

    Data = []
    Label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_{}*.h5'.format(partition))):
        with h5py.File(h5_name, 'r') as file:
            data = file['data'][:].astype('float32')
            label = file['label'][:].astype('int64')
            Data.append(data)
            Label.append(label)

    Data = np.concatenate(Data, axis=0)  # (9840, 2048, 3)
    Label = np.concatenate(Label, axis=0)  # (9840, 1)
    return Data, Label


def read_classed():
    with open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r') as file:
        shape_name = file.read()
        shape_name = np.array(shape_name.split('\n')[:-1])
        return shape_name


class ModelNet40(Dataset):
    def __init__(self, train=True, num_points=1024, randomize_data=False):
        super(ModelNet40, self).__init__()
        self.data, self.labels = load_data(train)
        self.shapes = read_classed()
        self.num_points = num_points
        self.randomize_data = randomize_data

    def __getitem__(self, index):
        if self.randomize_data:
            current_points = self.randomize(index)
        else:
            current_points = self.data[index].copy()

        current_points = torch.from_numpy(current_points).float()
        label = torch.from_numpy(self.labels[index]).type(torch.LongTensor)
        return current_points, label
    def __len__(self):
        return self.data.shape[0]

    def get_shape(self, label):
        return self.shapes[label]

    def randomize(self, index):
        point_index = np.arange(0, self.num_points)
        np.random.shuffle(point_index)
        return self.data[index, point_index].copy()


class RegistrationData(Dataset):
    def __init__(self, algorithm='GLFDQNet', data_class=ModelNet40(), is_testing=False):
        super(RegistrationData, self).__init__()
        self.algorithm = algorithm
        self.is_testing = is_testing
        self.data_class = data_class
        if self.algorithm == 'GLFDQNet':
            self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)

    def __getitem__(self, index):
        template_pc, label = self.data_class[index]
        self.transforms.index = index
        source_pc = self.transforms(template_pc)
        igt = self.transforms.igt
        if self.is_testing:
            return template_pc, source_pc, igt, self.transforms.igt_rotation, self.transforms.igt_translation
        else:
            return template_pc, source_pc, igt,self.transforms.igt_rotation, self.transforms.igt_translation

    def __len__(self):
        return len(self.data_class)


