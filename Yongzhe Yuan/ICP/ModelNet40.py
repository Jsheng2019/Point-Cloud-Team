from torch.utils.data import Dataset
import os
import glob
from read_off import getVertexsandMeshs
from torch.utils.data import DataLoader
import torchvision
import transforms

FILEPATH = r"E:\AI\Dataset\ModelNet40"

def get_classes(root):
    """  在{root}/路径下寻找所有类，并将所有类编号形成dict """
    classes = [modelclass for modelclass in os.listdir(root)]
    classes.sort()
    class_classid = {classes[i]: i for i in range(len(classes))}
    return classes, class_classid


def glob_dataset(root,class_classid,pattern):
    """
    :param root: 数据集根目录
    :param class_classid: 所有类和其编号的字典
    :param pattern: 训练模式或测试模式
    :return:  所有40类的样例的路径和编号  （path，id）
    """
    root=os.path.expanduser(root)
    samples=[]
    for target in sorted(os.listdir(root)):
        modelclass=os.path.join(root,target)
        target_idx=class_classid.get(target)     #通过key获取id
        names=glob.glob(os.path.join(modelclass,pattern))
        for path in sorted(names):
            samples.append((path,target_idx))
    return samples

class ModelNet40(Dataset):

    def __init__(self,rootdir,pattern='train/*.off',transform=None):
        super(ModelNet40, self).__init__()
        classes, class_classid=get_classes(rootdir)
        samples=glob_dataset(rootdir,class_classid,pattern)  #在pattern模式下的所有类别的样例的路径和编号
        self.rootdir=rootdir
        self.transform=transform
        self.pattern=pattern
        self.samples=samples
        self.classes=classes
        self.class_classid=class_classid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path,target=self.samples[index]
        sample=getVertexsandMeshs(path)       # 根据index找到实例，再根据实例所在路径获取点云具体信息。返回一个Mesh类的对象，类具体信息在read_off.py中
        if self.transform is not None:
            sample=self.transform(sample)     # 调用transform中的call函数，返回的是使用顶点坐标变换后的点云
        return sample,target

if __name__=='__main__':

    transform = torchvision.transforms.Compose([
        transforms.Mesh2Points(),
        transforms.OnUnitCube(),
    ])
    testdata=ModelNet40(rootdir=FILEPATH,pattern='test/*.off',transform=transform)

    testdataloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for i, data in enumerate(testdataloader):
        print(data)
        break
