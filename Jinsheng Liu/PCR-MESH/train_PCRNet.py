import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40 import RegistrationData, ModelNet40
# from model.Pointnet import PointNet choco  install ninja
from model.MeshNet import MeshNet
from model.iPCRNet import iPCRNet
from operations.transform_functions import PCRNetTransform
from losses.chamfer_distance import ChamferDistanceLoss
# from losses.earth_mover_distance import EMDLosspy
from losses.rmse_features import RMSEFeaturesLoss
from losses.frobenius_norm import FrobeniusNormLoss
import time
import transforms3d

BATCH_SIZE = 15
START_EPOCH = 0
MAX_EPOCHS = 100
#torch.cuda.set_device(1)
device = torch.device('cuda:0')
pretrained = ''  # 是否有训练过的模型可用
resume = ''  # 最新的检查点文件
exp_name = 'ipcrnet'


def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    count = 0
    duration = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt, R, T  = data
        # print(template[0].shape)
        # print(template[1].shape)

        for index in range(5):
            template[index] = template[index].to(device)  # [B,N,3]
            source[index] = source[index].to(device)  # [B,N,3]


        # source = source.to(device)  # [B,N,3]

        igt = igt.to(device)  # [B,1,7]
        R = R.to(device)
        T = T.to(device)
        for index in range(3):
            source[index] = source[index] - torch.mean(source[index], dim=2, keepdim=True)
            template[index] = template[index] - torch.mean(template[index], dim=2, keepdim=True)
        source[4] = source[4] - torch.mean(source[4], dim=1, keepdim=True)
        template[4] = template[4] - torch.mean(template[4], dim=1, keepdim=True)
        # source = source - torch.mean(source, dim=1, keepdim=True)
        # template = template - torch.mean(template, dim=1, keepdim=True)
        tic = time.time()
        output = model(device, template, source)


        # loss_val = ChamferDistanceLoss()(template, output['transformed_source'])   #对角损失
        # 7d转变换矩阵
        igt = igt.squeeze(1).contiguous()
        identity = torch.eye(3).to(device).view(1, 3, 3).expand(source[0].size(0), 3, 3).contiguous()
        est_R = PCRNetTransform.quaternion_rotate(identity, igt).permute(0, 2, 1)
        est_t = PCRNetTransform.get_translation(igt).view(-1, 1, 3)
        igt = PCRNetTransform.convert2transformation(est_R, est_t)

        #template_centers = template[0].permute(0, 2, 1).contiguous()
        template_corners = template[4]
        #print(est_R.shape, est_t.shape, igt.shape)
        #print(output[est_R], output[est_t])
        loss_val = ChamferDistanceLoss()(template_corners, output['transformed_source'])
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1
        toc = time.time()
        duration = duration + toc - tic

    train_loss = float(train_loss) / count
    duration = duration / count
    print('time is :',duration)
    return train_loss

def compute_accuracy(igt_R, pred_R, igt_t, pred_t):
    errors_temp = []
    for igt_R_i, pred_R_i, igt_t_i, pred_t_i in zip(igt_R, pred_R, igt_t, pred_t):
        errors_temp.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i))
    return np.mean(errors_temp, axis=0)

def find_errors(igt_R, pred_R, igt_t, pred_t):
    # igt_R:				Rotation matrix [3, 3] (source = igt_R * template)
    # pred_R: 			Registration algorithm's rotation matrix [3, 3] (template = pred_R * source)
    # igt_t:				translation vector [1, 3] (source = template + igt_t)
    # pred_t: 			Registration algorithm's translation matrix [1, 3] (template = source + pred_t)

    # Euler distance between ground truth translation and predicted translation.
    igt_t = -np.matmul(igt_R.T, igt_t.T).T			# gt translation vector (source -> template)
    translation_error = np.sqrt(np.sum(np.square(igt_t - pred_t)))

    # Convert matrix remains to axis angle representation and report the angle as rotation error.
    error_mat = np.dot(igt_R, pred_R)							# matrix remains [3, 3]
    _, angle = transforms3d.axangles.mat2axangle(error_mat)
    return translation_error, abs(angle*(180/np.pi))

def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    #loss1 = 0.00
    count = 0
    errors = []
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, R, T = data

        for index in range(5):
            template[index] = template[index].to(device)  # [B,N,3]
            source[index] = source[index].to(device)  # [B,N,3]
        igt = igt.to(device)
        R = R.to(device)
        T = T.to(device)

        # mean substraction
        for index in range(3):
            source[index] = source[index] - torch.mean(source[index], dim=2, keepdim=True)
            template[index] = template[index] - torch.mean(template[index], dim=2, keepdim=True)
        source[4] = source[4] - torch.mean(source[4], dim=1, keepdim=True)
        template[4] = template[4] - torch.mean(template[4], dim=1, keepdim=True)
        # source = source - torch.mean(source, dim=1, keepdim=True)
        # template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(device, template, source)
        est_R = output['est_R']
        est_t = output['est_t']
        est_T = output['est_T']

        errors.append(compute_accuracy(R.detach().cpu().numpy(), est_R.detach().cpu().numpy(),
                                       T.detach().cpu().numpy(), est_t.detach().cpu().numpy()))

        igt = igt.squeeze(1).contiguous()
        identity = torch.eye(3).to(device).view(1, 3, 3).expand(source[0].size(0), 3, 3).contiguous()
        est_R = PCRNetTransform.quaternion_rotate(identity, igt).permute(0, 2, 1)
        est_t = PCRNetTransform.get_translation(igt).view(-1, 1, 3)
        igt = PCRNetTransform.convert2transformation(est_R, est_t)


        #template_centers = template[0].permute(0, 2, 1).contiguous()
        template_corners = template[4]
        loss_val = ChamferDistanceLoss()(template_corners, output['transformed_source'])
        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss) / count
    errors = np.mean(np.array(errors), axis=0)
    #loss1 = float(loss1)/count
    #print('loss1:',loss1)
    return test_loss, errors[0], errors[1]


def train(model, train_loader, test_loader):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(learnable_params)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf
    for epoch in range(START_EPOCH, MAX_EPOCHS):
        train_loss = train_one_epoch(device, model, train_loader, optimizer)
        test_loss, translation_error, rotation_error = test_one_epoch(device, model, test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': optimizer.state_dict(), }
            torch.save(snap, 'checkpoints/{}/models2/best_model_snap.t7'.format(exp_name))
            torch.save(model.state_dict(), 'checkpoints/{}/models2/best_model.t7'.format(exp_name))
            torch.save(model.feature_model.state_dict(), 'checkpoints/{}/models2/best_ptnet_model.t7'.format(exp_name))

        torch.save(snap, 'checkpoints/{}/models2/model_snap.t7'.format(exp_name))
        torch.save(model.state_dict(), 'checkpoints/{}/models2/model.t7'.format(exp_name))
        torch.save(model.feature_model.state_dict(), 'checkpoints/{}/models2/ptnet_model.t7'.format(exp_name))

        print("EPOCH:{},Training Loss:{},Testing Loss:{},Best Loss:{}".format(epoch + 1, train_loss, test_loss,
                                                                              best_test_loss))
        print("Rotation Error: {} & Translation Error: ".format(rotation_error, translation_error))
        print(" ")


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    #返回列表：模板点云，源点云，真实的变换矩阵7d，真实的旋转矩阵，真实的平移向量
    trainset = RegistrationData('PCRNet', ModelNet40(part='train'), is_testing=False)
    testset = RegistrationData('PCRNet', ModelNet40(part='test'), is_testing=True)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    # ptnet = PointNet(emb_dims=1024)
    model = iPCRNet(feature_model=MeshNet)
    model = model.to(device)

    checkpoint = None
    if resume:
        checkpoint = torch.load(resume)
        START_EPOCH = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))

    model.to(device)

    train(model, trainloader, testloader)
