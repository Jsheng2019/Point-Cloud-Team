import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MeshNet import MeshNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform


class iPCRNet(nn.Module):
    def __init__(self, feature_model=MeshNet, droput=0.0, pooling='max'):
        super(iPCRNet, self).__init__()
        self.feature_model = feature_model()
        self.pooling = Pooling(pooling)

        self.fc1 = nn.Linear(2048, 1024)
        #使用Pointnet池化后应该是1*1152

        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)

        #无dropout
        self.fc6 = nn.Linear(256, 7)

    def forward(self, device, template, source, max_iteration=1):
        #估计的旋转矩阵
        est_R = torch.eye(3).to(device).view(1, 3, 3).expand(template[0].size(0), 3, 3).contiguous()  # (Bx3x3)
        #估计的平移向量
        est_t = torch.zeros(1, 3).to(device).view(1, 1, 3).expand(template[0].size(0), 1, 3).contiguous()  # (Bx1x3)

        centers, corners, normals, neighbor_index = template[0], template[1], template[2], template[3]

        template_features = self.feature_model(centers, corners, normals, neighbor_index)   #模板点云的全局特征

        if max_iteration == 1:
            est_R, est_t, source_corners = self.spam(device, template_features, source, est_R, est_t)
        else:
            for i in range(max_iteration):
                est_R, est_t, source_corners = self.spam(template_features, source, est_R, est_t)



        result = {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                  'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),  # source -> template   #得到估计的变换矩阵[B,4,4]
                  'r': template_features - self.source_features,    #得到两个全局特征的差值[B,feature_shape]
                  'transformed_source': source_corners}        #变换后最终的源点云
        return result

    def spam(self, device, template_features, source, est_R, est_t):
        batch_size = source[0].size(0)
        centers_trans, corners_trans, normals_trans, neighbor_index, corners_cat_trans = source[0], source[1], source[2], source[3], source[4]

        self.source_features = self.feature_model(centers_trans, corners_trans, normals_trans, neighbor_index)   #源点云的全局特征

        y = torch.cat([template_features, self.source_features], dim=1)

        pose_7d = F.relu(self.fc1(y))
        pose_7d = F.relu(self.fc2(pose_7d))
        pose_7d = F.relu(self.fc3(pose_7d))
        pose_7d = F.relu(self.fc4(pose_7d))
        pose_7d = F.relu(self.fc5(pose_7d))
        pose_7d = self.fc6(pose_7d)      #[B,7]
        #得到[R,t]的7维向量
        pose_7d = PCRNetTransform.create_pose_7d(pose_7d)   #[B,7]

        identity = torch.eye(3).to(device).view(1, 3, 3).expand(source[0].size(0), 3, 3).contiguous()  #[B,3,3]

        est_R_temp = PCRNetTransform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
        est_t_temp = PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3)

        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp   #[B,1,3]
        est_R = torch.bmm(est_R_temp, est_R)     #[B,3,3]

        #将角点值合并成[B,n,3],并去重
        #corners_trans = corners_trans.permute(0,2,1).contiguous()
        # corners1 = corners_trans[:, :, 0:2]
        # corners2 = corners_trans[:, :, 3:5]
        # corners2 = corners_trans[:, :, 6:8]


        #corners_trans =
        #centers_trans = centers_trans.permute(0,2,1).contiguous()
        #print(len(centers_trans[0]))
        #source_centers = PCRNetTransform.quaternion_transform(centers_trans, pose_7d)  # Ps' = est_R*Ps + est_t   #更新后的源点云
        corners_cat_trans = PCRNetTransform.quaternion_transform(corners_cat_trans, pose_7d)
        #返回列表：估计的旋转矩阵，估计的平移向量，变换后的源点云
        # return est_R, est_t, source
        return est_R, est_t, corners_cat_trans


if __name__ == '__main__':
    template, source = torch.rand(10, 1024, 3), torch.rand(10, 1024, 3)
    pn = PointNet()

    net = iPCRNet(pn)

    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
    print(result['est_T'].shape)
    print(result['r'].shape)
    print(result['transformed_source'].shape)
    print('-'*50)
