
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40 import RegistrationData,ModelNet40
from model.GLFNet import PointNet
from model.GLFDQNet import GLFDQNet
from losses.chamfer_distance import ChamferDistanceLoss
from losses.frobenius_norm import FrobeniusNormLoss
from operations.transform_functions import PCRNetTransform

BATCH_SIZE=32
EVAL=False
START_EPOCH=0
MAX_EPOCHS=200
pretrained=''

def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0

    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, igt_R, igt_t = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)
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
    test_loss = float(test_loss)/count
    return test_loss

if __name__ == '__main__':
    testset = RegistrationData('GLFDQNet', ModelNet40(train=False),is_testing=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
    device = torch.device('cpu')

    ptnet = PointNet(emb_dims=1024)
    model = GLFDQNet(feature_model=ptnet)
    model = model.to(device)

    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))

    model.to(device)
    test_loss = test_one_epoch(device, model, testloader)
    print("Test Loss: {}".format(test_loss))
