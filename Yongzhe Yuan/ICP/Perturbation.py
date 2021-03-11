from torch.utils.data import Dataset
from ModelNet40 import ModelNet40
from compute_utils import se3
from compute_utils import so3
import torch
import numpy as np


#  wv 表示扭曲（默认） wt表示旋转和平移
class Perturbation(Dataset):
    def __init__(self,dataset,perturbation,source_modifier=None,template_modifier=None,format_translate=False):
        self.dataset=dataset
        self.perturbation=np.array(perturbation)
        self.source_modifier=source_modifier
        self.template_modifier=template_modifier
        self.format_translate=format_translate     #False表示wv，True代表wt

    def __len__(self):
        return len(self.dataset)

    def do_transform(self,p0,x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.format_translate:
            # x: 扭曲向量
            g=se3.exp(x).to(p0)   # [1, 4, 4]
            p1=se3.transform(g,p0)
            igt=g.squeeze(0)
        else:
            # x: 旋转和平移
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0)  # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R  # 旋转
            g[:, 0:3, 3] = q  # 平移
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0)  # igt: p0 -> p1
        return p1, igt

    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):  # 扰动
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):  # 旋转
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __getitem__(self, index):
        twist = torch.from_numpy(np.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm
        # p0: 模板点云, p1: 源点云, igt: 从p0变为p1的变换矩阵
        return p0, p1, igt


if __name__ == '__main__':
    FILEPATH = r"E:\AI\Dataset\ModelNet40"
    testdata = ModelNet40(rootdir=FILEPATH, pattern='train/*.off', transform=None)
    testdata = Perturbation(testdata,perturbation=None,format_translate=False)
