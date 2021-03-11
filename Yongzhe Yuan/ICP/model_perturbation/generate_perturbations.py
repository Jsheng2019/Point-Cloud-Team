import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from Perturbation import Perturbation
import torchvision
import transforms
from ModelNet40 import ModelNet40
import numpy as np

rootdir = r"E:\AI\Dataset\ModelNet40"
Translation = True  # 是否平移
transform = torchvision.transforms.Compose([
    transforms.Mesh2Points(),
    transforms.OnUnitCube(),
])

for mag in range(16):
    testdata = ModelNet40(rootdir=rootdir, pattern='test/*.off', transform=transform)
    if Translation:
        x = Perturbation.generate_perturbations(len(testdata), mag / 10, randomly=False)
    else:
        x = Perturbation.generate_rotations(len(testdata), mag / 10, randomly=False)
    print(mag / 10)
    np.savetxt("pert_{}.csv".format(mag), x, delimiter=',')
