import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix,mean_squared_error
from skimage.metrics import structural_similarity as ssim
from utils import save_imgs
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch_SH(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images = data[:, :5, :, :, :]
        targets = data[:, 5:, :, :, :]
        # images = images.unsqueeze(1)

        images, targets = images.to(device).float(), targets.to(device).float()
        targets=targets.squeeze(2)

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out= model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # if iter % config.print_interval == 0:
        #     log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
        #     print('')
        #     print(log_info)
        # #     logger.info(log_info)

    train_loss=np.mean(loss_list)
    train_loss=round(train_loss,5)
    scheduler.step()
    log_info = f'Train_loss: {train_loss:.5f}'
    print(log_info)
    logger.info(log_info)
    return train_loss


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images = data[:, :5, :, :]
        targets = data[:, 5:, :, :]


        images, targets = images.to(device).float(), targets.to(device).float()


        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # if iter % config.print_interval == 0:
        #     log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
        #     print('')
        #     print(log_info)
        # #     logger.info(log_info)

    train_loss = np.mean(loss_list)
    train_loss = round(train_loss, 5)
    scheduler.step()
    log_info = f'Train_loss: {train_loss:.5f}'
    print(log_info)
    logger.info(log_info)
    return train_loss

def val_one_epoch_SH(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in test_loader:
            img = data[:, :5, :, :, :]
            msk = data[:, 5:, :, :, :]
            # img = img.unsqueeze(1)

            img, msk = img.to(device).float(), msk.to(device).float()
            msk = msk.squeeze(2)
            out= model(img)
            loss = criterion(out, msk)
            # out = model(img)
            # loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)
    
    return np.mean(loss_list)


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in test_loader:
            img = data[:, :5, :, :]
            msk = data[:, 5:, :, :]
            # img = img.unsqueeze(1)

            img, msk = img.to(device).float(), msk.to(device).float()

            out = model(img)
            loss = criterion(out, msk)
            # out = model(img)
            # loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        for threshold in config.threshold:
            y_pre = np.where(preds >= threshold, 1, 0)
            y_true = np.where(gts >= threshold, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            HSS = (float(TP) * float(TN) - float(FN) * float(FP)) / (
                        ((float(TP) + float(FN)) * ((float(FN) + float(TN)))) + (
                            (float(TP) + float(FP)) * ((float(FP) + float(TN)))))
            POD = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            CSI = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            # print('')
            log_info = f'{threshold}:,val epoch: {epoch}, loss: {np.mean(loss_list):.5f},accuracy: {accuracy:.4f},CSI: {CSI:.4f}, HSS:{HSS:.4f}, POD: {POD:.4f}'
            # print(log_info)
            logger.info(log_info)

    else:
        # print('')
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate((test_loader)):
            img = data[:, :5, :, :]
            msk = data[:, 5:, :, :]
            # img = img.unsqueeze(1)

            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out= model(img)
            loss = criterion(out, msk)

            # out = model(img)
            # loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out) 
            # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        for threshold in config.threshold:
            y_pre = np.where(preds>=threshold, 1, 0)
            y_true = np.where(gts>=threshold, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            HSS = (float(TP) * float(TN) - float(FN) * float(FP)) / (
                        ((float(TP) + float(FN)) * ((float(FN) + float(TN)))) + (
                            (float(TP) + float(FP)) * ((float(FP) + float(TN)))))
            POD = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            CSI = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            # 计算均方根误差 (RMSE)
            RMSE = np.sqrt(mean_squared_error(gts, preds))

            # 计算虚警率 (False Alarm Ratio, FAR)
            FAR = float(FP) / float(TP + FP) if float(TP + FP) != 0 else 0

            # 计算结构相似性 (SSIM)
            SSIM = ssim(gts.reshape(-1), preds.reshape(-1), data_range=1)  # 假设 gts 和 preds 的形状相同



            if test_data_name is not None:
                log_info = f'test_datasets_name: {test_data_name}'
                print(log_info)
                logger.info(log_info)
            log_info = f'{threshold}: CSI: {CSI:.4f}, HSS:{HSS:.4f}, POD: {POD:.4f}, FAR: {FAR:.4f}, SSIM: {SSIM:.4f}, RMSE:{RMSE:.4f}'
            print(log_info)
            logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch_SH(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    from dataset.metrics import SimplifiedEvaluator
    evaluator = SimplifiedEvaluator(
        seq_len=20,
        value_scale=90,
        thresholds=[20, 30, 35, 40]
    )
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate((test_loader)):
            # img = data[:, :5, :, :]
            # msk = data[:, 5:, :, :]



            img = data[:, :5, :, :, :]
            msk = data[:, 5:, :, :, :]


            # img = img.unsqueeze(1)

            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            msk = msk.squeeze(2)
            # img, msk = img.to(device).float(), msk.to(device).float()
            out= model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(2).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)
            # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)





######################shanghai#####################
        preds = np.array(preds)
        # print(preds.shape)

        preds = preds.reshape(-1, preds.shape[2], preds.shape[3], preds.shape[4]).astype(np.float32)
        gts = np.array(gts)
        # print(gts.shape)
        gts = gts.reshape(-1, gts.shape[2], gts.shape[3], gts.shape[4]).astype(np.float32)
        # print(preds.shape)
        # print(gts.shape)
        evaluator.evaluate(preds, gts)
        results = evaluator.done()
        for thresh, metrics in results["threshold_metrics"].items():
            print(f"{thresh}mm CSI: {metrics['CSI']:.4f} POD: {metrics['POD']:.4f} HSS: {metrics['HSS']:.4f}")
        print("\nOverall Metrics:")
        print(f"FAR:  {results['FAR']:.4f}")
        print(f"RMSE: {results['RMSE']:.2f}")
        print(f"SSIM: {results['SSIM']:.4f}")
        # print(f"LPIPS: {results['LPIPS']:.4f}")
    return np.mean(loss_list)
