import torch.nn as nn
import torch
import torch.nn.functional as F
from operations.Pooling import Pooling

class LFU(nn.Module):
    def __init__(self, input_dim, pooling='max'):
        super(LFU, self).__init__()
        self.pooling = Pooling(pooling)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, input_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # [B,N,1024]
        x = self.pooling(x)
        return x


class PointNet(nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc"):
        # emb_dims:			Embedding Dimensions for PointNet.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PointNet, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        self.input_shape = input_shape
        self.emb_dims = emb_dims

        self.lfu1 = LFU(input_dim=64)
        self.lfu2 = LFU(input_dim=64)
        self.lfu3 = LFU(input_dim=64)
        self.lfu4 = LFU(input_dim=128)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64 + 64, 64, 1)
        self.conv4 = nn.Conv1d(64 +64, 128, 1)
        self.conv5 = nn.Conv1d(128 + 64, self.emb_dims, 1)

    def forward(self, input_data):
        # input_data: 		Point Cloud having shape input_shape.
        # output:			PointNet features (Batch x emb_dims)
        if self.input_shape == "bnc":
            num_points = input_data.shape[1]
            input_data = input_data.permute(0, 2, 1)
        else:
            num_points = input_data.shape[2]

        output1 = F.relu(self.conv1(input_data))  # [B,64,N]

        output2 = F.relu(self.conv2(output1))  # [B,64,N]
        lfu1 = self.lfu1(output1).unsqueeze(1).contiguous()  # [B,1,64]
        lfu1 = lfu1.repeat(1, num_points, 1).permute(0,2,1)   # [B,1,64]->[B,N,64]->[B,64,N]
        output = torch.cat([output2, lfu1], dim=1)  # [B,128,N]


        output3 = F.relu(self.conv3(output))  # [B,64,N]
        lfu2 = self.lfu2(output2).unsqueeze(1).contiguous()  # [B,1,64]
        lfu2 = lfu2.repeat(1, num_points, 1).permute(0,2,1)  # [B,1,64]->[B,N,64]->[B,64,N]
        output = torch.cat([output3, lfu2], dim=1)  # [B,128,N]

        output4 = F.relu(self.conv4(output))  # [B,128,N]
        lfu3 = self.lfu3(output3).unsqueeze(1).contiguous()  # [B,1,64]
        lfu3 = lfu3.repeat(1, num_points, 1).permute(0,2,1)  # [B,1,64]->[B,N,64]->[B,64,N]
        output = torch.cat([output4, lfu3], dim=1)  # [B,192,N]

        output5 = F.relu(self.conv5(output))  # [B,1024,N]
        lfu4 = self.lfu4(output4).unsqueeze(1).contiguous()  # [B,1,128]
        lfu4 = lfu4.repeat(1, num_points, 1).permute(0,2,1)  # [B,1,128]->[B,N,128]->[B,128,N]
        output = torch.cat([output5, lfu4], dim=1)  # [B,1152,N]

        return output



