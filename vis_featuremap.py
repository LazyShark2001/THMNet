import os
import cv2
import cmapy
import time
import argparse
import numpy as np
import torch.autograd
from matplotlib import pyplot as plt
from skimage import io
from skimage.exposure import rescale_intensity
import torchvision.transforms as transforms
from torchvision.transforms import functional as transF
from collections import OrderedDict
from PIL import Image
from data import get_loader
from model.DCPNet import DCPNet
import torch.nn.functional as F

################## Model ##################

NET_NAME = 'MyNet'
DATA_NAME = 'Hot_Map'


class PredOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--T', required=False, default=3.0, help='Test time augmentation')
        parser.add_argument('--test_dir', required=False, default=os.path.join('Visualization', 'image'),
                            help='directory to test images')
        parser.add_argument('--pred_dir', required=False,
                            default=os.path.join(working_path, 'Visualization', DATA_NAME, NET_NAME),
                            help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False, default='EORSSD.pth')
        parser.add_argument('--dev_id', required=False, default=0, help='Device id')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0],
            [0, 0, 128]]

preprocess = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def main():
    begin_time = time.time()
    opt = PredOptions().parse()

    net = DCPNet()
    state_dict = torch.load(opt.chkpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
    net.load_state_dict(new_state_dict)
    net.to(torch.device('cuda', int(opt.dev_id))).eval()

    predict(net, opt)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


def predict(net, opt):
    imgA_dir = opt.test_dir

    if not os.path.exists(opt.pred_dir): os.makedirs(opt.pred_dir)
    pred_mA_dir = os.path.join(opt.pred_dir)
    if not os.path.exists(pred_mA_dir): os.makedirs(pred_mA_dir)
    data_list = os.listdir(imgA_dir)
    valid_list = []
    for it in data_list:
        if it[-4:] == '.jpg': valid_list.append(it)

    for it in valid_list:
        imgA_path = os.path.join(imgA_dir, it)
        imgA = io.imread(imgA_path)
        imgA_PIL = Image.fromarray(imgA)
        imgA_pre = preprocess(imgA_PIL)
        with torch.no_grad():
            tensorA = imgA_pre.unsqueeze(0).to(torch.device('cuda', int(opt.dev_id))).float()
            sal, sig_sal,att = net(tensorA)

            att = F.upsample(att, size=imgA.shape[0:2], mode='bilinear', align_corners=True)

            att_map = att.squeeze(0).detach().cpu().numpy()
            # att = att.data.cpu().numpy()
            latent_num = att_map.shape[0]
            for idx in range(latent_num):
                latentA = rescale_intensity(att_map[idx], out_range=(0, 255)).astype(np.uint8)
                latentA_color = cv2.applyColorMap(latentA, cmapy.cmap('jet')) #加注意力
                pred_pathA = os.path.join(pred_mA_dir, it[:-4] + '_' + str(idx) + '.png')
                io.imsave(pred_pathA, latentA_color)
                # print(idx)
                # plt.imshow(att[0][idx], cmap=plt.cm.get_cmap('jet'))
                # plt.imsave(str(pred_mA_dir) + str(it[:-4]) + '_' + str(idx) + '.png', att[0][idx],
                #            cmap=plt.cm.get_cmap('jet'))
                # plt.pause(0.1)


if __name__ == '__main__':
    main()
