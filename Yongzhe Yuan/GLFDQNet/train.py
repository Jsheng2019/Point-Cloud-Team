import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40 import RegistrationData, ModelNet40
from model.GLFNet import PointNet
from model.GLFDQNet import GLFDQNet
from operations.transform_functions import PCRNetTransform
from losses.chamfer_distance import ChamferDistanceLoss
from losses.frobenius_norm import FrobeniusNormLoss

torch.cuda.set_device(1)
BATCH_SIZE = 32
START_EPOCH = 0
MAX_EPOCHS = 200
device = torch.device('cuda:1')
pretrained = ''
resume = ''
exp_name = 'glfdqnet'


def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt, R, T = data
        template = template.to(device)  # [B,N,3]
        source = source.to(device)  # [B,N,3]
        igt = igt.to(device)  # [B,1,7]
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)
        output = model(template, source)


        igt = igt.squeeze(1).contiguous()
        identity = torch.eye(3).to(source).view(1, 3, 3).expand(source.size(0), 3, 3).contiguous()
        est_R = PCRNetTransform.quaternion_rotate(identity, igt).permute(0, 2, 1)
        est_t = PCRNetTransform.get_translation(igt).view(-1, 1, 3)
        igt = PCRNetTransform.convert2transformation(est_R, est_t)

        loss_val = ChamferDistanceLoss()(template, output['transformed_source']) + 0.007 * FrobeniusNormLoss()(
            output['est_T'], igt)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1

    train_loss = float(train_loss) / count
    return train_loss


def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, R, T = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        # mean substraction
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(template, source)

        igt = igt.squeeze(1).contiguous()
        identity = torch.eye(3).to(source).view(1, 3, 3).expand(source.size(0), 3, 3).contiguous()
        est_R = PCRNetTransform.quaternion_rotate(identity, igt).permute(0, 2, 1)
        est_t = PCRNetTransform.get_translation(igt).view(-1, 1, 3)
        igt = PCRNetTransform.convert2transformation(est_R, est_t)
        loss_val = ChamferDistanceLoss()(template, output['transformed_source']) + 0.007 * FrobeniusNormLoss()(
            output['est_T'], igt)

        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss) / count
    return test_loss


def train(model, train_loader, test_loader):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(learnable_params)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf
    for epoch in range(START_EPOCH, MAX_EPOCHS):
        train_loss = train_one_epoch(device, model, train_loader, optimizer)
        test_loss = test_one_epoch(device, model, test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': optimizer.state_dict(), }
            torch.save(snap, 'checkpoints/{}/models/best_model_snap.t7'.format(exp_name))
            torch.save(model.state_dict(), 'checkpoints/{}/models/best_model.t7'.format(exp_name))
            torch.save(model.feature_model.state_dict(), 'checkpoints/{}/models/best_ptnet_model.t7'.format(exp_name))

        torch.save(snap, 'checkpoints/{}/models/model_snap.t7'.format(exp_name))
        torch.save(model.state_dict(), 'checkpoints/{}/models/model.t7'.format(exp_name))
        torch.save(model.feature_model.state_dict(), 'checkpoints/{}/models/ptnet_model.t7'.format(exp_name))

        print("EPOCH:{},Training Loss:{},Testing Loss:{},Best Loss:{}".format(epoch + 1, train_loss, test_loss,
                                                                              best_test_loss))


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    trainset = RegistrationData('GLFDQNet', ModelNet40(train=True))
    testset = RegistrationData('GLFDQNet', ModelNet40(train=False))
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    ptnet = PointNet(emb_dims=1024)
    model = GLFDQNet(feature_model=ptnet)
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
