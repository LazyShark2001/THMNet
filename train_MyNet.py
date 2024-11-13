import argparse
import os
import time

from torch.utils.tensorboard import SummaryWriter

import pytorch_iou
from Eval.eval import SOD_Eval
# from data import get_loader
from data import get_loader
from model.MyNet import Net
# from model.baseline import Net
import datetime

from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(torch.cuda.get_device_name(0))
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')  # 352 384 512
parser.add_argument('--val_interval', type=int, default=1, help='validation interval ')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=45, help='every n epochs decay learning rate')
# dataset
parser.add_argument('--data_path', type=str, default='SOD_DataSet/EORSSD_aug', help='DataSet Path')
parser.add_argument('--root', type=str, default='Net_MSA_Decoder', help='project root')

opt = parser.parse_args()

# build models
model = Net()
model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

train_image_root = opt.data_path+'/train/image/'
train_gt_root = opt.data_path+'/train/GT/'


train_loader = get_loader(train_image_root, train_gt_root, batchsize=opt.batchsize, size=opt.trainsize,
                          is_train=True)

train_total_step = len(train_loader)

print(train_total_step)

# loss
bce_loss = torch.nn.BCEWithLogitsLoss()
iou_loss = pytorch_iou.IOU(size_average=True)

def train_one_epoch(train_loader, model, optimizer, epoch):
    model.train()
    mean_loss = []
    for i, pack in enumerate(train_loader, start=1):

        images, gts = pack
        images = images.cuda()
        gts = gts.cuda()
       
        P5, P5_sig, P4, P4_sig, P3, P3_sig, P2, P2_sig, P1, P1_sig = model(images)
        # P4, P4_sig, P3, P3_sig, P2, P2_sig, P1, P1_sig = model(images)
        # 原损失
        # loss_ce = bce_loss(sal, gts)
        # loss_iou = iou_loss(sal_sig, gts)

        loss5 = bce_loss(P5, gts) + iou_loss(P5_sig, gts)
        loss4 = bce_loss(P4, gts) + iou_loss(P4_sig, gts)
        loss3 = bce_loss(P3, gts) + iou_loss(P3_sig, gts)
        loss2 = bce_loss(P2, gts) + iou_loss(P2_sig, gts)
        loss1 = bce_loss(P1, gts) + iou_loss(P1_sig, gts)

        loss = loss1+loss2+loss3+loss4+loss5
        # loss = loss1+loss2+loss3+loss4

        optimizer.zero_grad()
        
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        mean_loss.append(loss.data)
        if i % 20 == 0 or i == train_total_step:
            print(
                'Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}'.
                format(epoch, opt.epoch, i, train_total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data))

    train_mean_loss = sum(mean_loss) / len(mean_loss)
    return train_mean_loss, opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch)


print("strart train")
if __name__ == '__main__':

    current_Sm = 0.0
    time_sum = 0
    # 添加tensorboard
    writer = SummaryWriter(opt.root+"/logs_train")  # tensorboard --logdir=logs_train

    start_time = time.time()
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        time_start = time.time()
        train_mean_loss, lr = train_one_epoch(train_loader, model, optimizer, epoch)
        time_end = time.time()
        time_sum = int(time_end - time_start)
        print(f"训练一轮需要的时间: {time_sum} 秒")
        # save weight
        save_path = opt.root+'/result/weight/'
        datasat=opt.data_path.split('/')[-1].split('_')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % 1 == 0:
            torch.save(model.state_dict(), save_path + datasat+'.pth' + '.%d' % epoch,
                       _use_new_zipfile_serialization=False)

        # evaluate
        if epoch % opt.val_interval == 0:
            Sm_info, MAE_info, maxEm_info, maxFm_info = SOD_Eval(epoch,opt.data_path,opt.root)

            # 可视化
            writer.add_scalar("train_loss", train_mean_loss, epoch)
            writer.add_scalar("val_Sm", Sm_info, epoch)
            writer.add_scalar("val_mae", MAE_info, epoch)
            writer.add_scalar("val_maxEm", maxEm_info, epoch)
            writer.add_scalar("val_maxFm", maxFm_info, epoch)

            # save_best
            if current_Sm <= Sm_info:
                torch.save(model.state_dict(), opt.root+'/result/weight/' + datasat+'_best.pth' + '.%d' % epoch,
                           _use_new_zipfile_serialization=False)
                current_Sm = Sm_info

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
