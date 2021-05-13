import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GLFNet import PointNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform
from operations.dual import dual_quat_to_extrinsic

class GLFDQNet(nn.Module):
    def __init__(self, feature_model=PointNet(), pooling='max'):
        super(GLFDQNet, self).__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)

        self.fc1 = nn.Linear(1152 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)

        self.fc6 = nn.Linear(256, 8)

    def forward(self, template, source, max_iteration=2):
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        template_features = self.pooling(self.feature_model(template))


        if max_iteration == 1:
            est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
        else:
            for i in range(max_iteration):
                est_R, est_t, source = self.spam(template_features, source, est_R, est_t)

        result = {'est_R': est_R,  # source -> template [B,3,3]
                  'est_t': est_t,  # source -> template [B,1,3]
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),  # source -> template   #[B,4,4]
                  'r': template_features - self.source_features,
                  'transformed_source': source}
        return result

    def spam(self, template_features, source, est_R, est_t):
        self.source_features = self.pooling(self.feature_model(source))

        y = torch.cat([template_features, self.source_features], dim=1)

        pose_8d = F.relu(self.fc1(y))
        pose_8d = F.relu(self.fc2(pose_8d))
        pose_8d = F.relu(self.fc3(pose_8d))
        pose_8d = F.relu(self.fc4(pose_8d))
        pose_8d = F.relu(self.fc5(pose_8d))
        pose_8d = self.fc6(pose_8d)     #[B,8]

        pose_8d = PCRNetTransform.create_pose_8d(pose_8d)   #[B,8]

        R_qe = pose_8d[:, 0:4]
        D_qe = pose_8d[:, 4:]
        # Find current rotation and translation.
        est_R_temp, est_t_temp = dual_quat_to_extrinsic(R_qe, D_qe)
        # update translation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        # update rotation matrix.
        est_R = torch.bmm(est_R_temp, est_R)

        source = PCRNetTransform.quaternion_transform2(source, pose_8d, est_t_temp)  # Ps' = est_R*Ps + est_t

        return est_R, est_t, source

