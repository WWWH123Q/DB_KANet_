import random

import torch
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from torch.utils.data import DataLoader
# from loader import *
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader


from models.DB_KANet import  DB_KANet
from engine import *
import os
import sys
#................
# python -m trainvm2
import torch.optim.optimizer
import torchvision
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import h5py
import numpy as np
import torch.optim.optimizer
import torchvision

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# torch.autograd.set_detect_anomaly(True)
from torch.nn import SyncBatchNorm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,mean_squared_error
from skimage.metrics import structural_similarity as ssim
import os
import sys
import pickle
import subprocess
#................
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')

    resume_model = os.path.join('results/Depthconv__29_November_14h_11m_42s/checkpoints','best-epoch28-loss21847.0173.pth')
    # resume_model = os.path.join('results/mamba+mamba_loss_0.0001__22_November_00h_21m_37s/checkpoints', 'best.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')


    dataset = 'Shanghai'
    if dataset == 'Shanghai':
        from dataset.Shanghai import train_dataset, val_dataset, test_dataset
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    else :
        with h5py.File(r'', 'r') as hf:
            data = hf['vil'][:]

        num_samples = int(data.shape[0])
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        group_size = 8

        num_train = int(train_ratio * (num_samples - group_size + 1))
        num_train = int(train_ratio * (num_samples - group_size + 1))
        num_val = int(val_ratio * (num_samples - group_size + 1))
        num_test = (num_samples - group_size + 1) - num_train - num_val
        groups = [data[i:i + group_size] for i in range(0, num_samples - group_size)]

        # 划分训练集、验证集和测试集
        train_groups = groups[:num_train]
        valid_groups = groups[num_train:num_val + num_train]
        test_groups = groups[num_train + num_val:]
        train_groups = torch.tensor(np.array(train_groups))
        valid_groups = torch.tensor(np.array(valid_groups))
        test_groups = torch.tensor(np.array(test_groups))

        train_loader = DataLoader(train_groups, config.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(valid_groups, config.batch_size, drop_last=True)
        test_loader = DataLoader(test_groups, config.batch_size, drop_last=True)




    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = DB_KANet(num_classes=model_cfg['num_classes'],
                               input_channels=model_cfg['input_channels'],
                               c_list=model_cfg['c_list'],
                               c_list1=model_cfg['c_list1'],
                               split_att=model_cfg['split_att'],
                               bridge=model_cfg['bridge'], )

    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()


    print('#----------Set other params----------#')
    min_loss = 9999999999
    start_epoch = 1
    min_epoch = 1
    earlystopping_count=0
    earlystopping=10

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    total_params = sum(p.numel() for p in model.parameters())
    print("Number of Model Parameters:", total_params)
    print('#----------Training----------#')
    loss1_list = []
    loss2_list = []
    for epoch in tqdm(range(start_epoch, config.epochs + 1)):

        torch.cuda.empty_cache()
        if dataset == 'Shanghai':
            loss1 = train_one_epoch_SH(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                logger,
                config,
                scaler=scaler
            )
        else:
            loss1 = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                logger,
                config,
                scaler=scaler
            )
        loss1_list.append(loss1)
        if dataset == 'Shanghai':
            loss = val_one_epoch_SH(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
        else:
            loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
        loss2_list.append(loss)
        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'UltraLight.pth'))


    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)

        if dataset=='Shanghai':
            loss = test_one_epoch_SH(
                test_loader,
                model,
                criterion,
                logger,
                config,
            )
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            )
        else:
            loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
            )
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            )

    return loss1_list

if __name__ == '__main__':
    config = setting_config
    loss1_list=main(config)
    plt.plot(loss1_list,label='Train_loss')
    plt.title('VM Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()