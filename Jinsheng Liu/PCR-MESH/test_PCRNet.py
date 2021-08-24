import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40 import RegistrationData,ModelNet40
#from model.Pointnet import PointNet
from model.iPCRNet import iPCRNet
from model.MeshNet import MeshNet
from losses.chamfer_distance import ChamferDistanceLoss
import transforms3d
from losses.rmse_features import RMSEFeaturesLoss
from losses.frobenius_norm import FrobeniusNormLoss
from operations.transform_functions import PCRNetTransform

BATCH_SIZE=20
EVAL=False
START_EPOCH=0
MAX_EPOCHS=200
pretrained='checkpoints/ipcrnet/models2/best_model.t7'       #使用最好的模型参数测试


# Find error metrics.
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

def compute_accuracy(igt_R, pred_R, igt_t, pred_t):
    errors_temp = []
    for igt_R_i, pred_R_i, igt_t_i, pred_t_i in zip(igt_R, pred_R, igt_t, pred_t):
        errors_temp.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i))
    return np.mean(errors_temp, axis=0)

def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0
    errors = []

    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, igt_R, igt_t = data

        for index in range(5):
            template[index] = template[index].to(device)  # [B,N,3]
            source[index] = source[index].to(device)  # [B,N,3]

        # template = template.to(device)
        # source = source.to(device)
        igt = igt.to(device)

        # source_original = source.clone()
        # template_original = template.clone()
        igt_t = igt_t - torch.mean(source[4], dim=1).unsqueeze(1)
        # source[4] = source[4] - torch.mean(source[4], dim=1, keepdim=True)
        # template[4] = template[4] - torch.mean(template[4], dim=1, keepdim=True)
        for index in range(3):
            source[index] = source[index] - torch.mean(source[index], dim=2, keepdim=True)
            template[index] = template[index] - torch.mean(template[index], dim=2, keepdim=True)
        source[4] = source[4] - torch.mean(source[4], dim=1, keepdim=True)
        template[4] = template[4] - torch.mean(template[4], dim=1, keepdim=True)

        output = model(device, template, source)
        est_R = output['est_R']
        est_t = output['est_t']
        est_T = output['est_T']

        errors.append(compute_accuracy(igt_R.detach().cpu().numpy(), est_R.detach().cpu().numpy(),
                                       igt_t.detach().cpu().numpy(), est_t.detach().cpu().numpy()))

        # transformed_source = torch.bmm(est_R, source.permute(0, 2, 1)).permute(0,2,1) + est_t
        loss_val = ChamferDistanceLoss()(template[4], output['transformed_source'])

        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss)/count
    errors = np.mean(np.array(errors), axis=0)
    return test_loss, errors[0], errors[1]

if __name__ == '__main__':
    testset = RegistrationData('PCRNet', ModelNet40(part='test'),is_testing=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
    device = torch.device('cpu')

    #ptnet = PointNet(emb_dims=1024)
    model = iPCRNet(feature_model=MeshNet)
    model = model.to(device)

    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))

    model.to(device)
    test_loss, translation_error, rotation_error = test_one_epoch(device, model, testloader)
    print("Test Loss: {}, Rotation Error: {} & Translation Error: {}".format(test_loss, rotation_error,translation_error))
