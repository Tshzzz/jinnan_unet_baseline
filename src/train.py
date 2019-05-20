import os
import errno
import tqdm
import numpy as np
import random

from src.utils import  txt_logger
from src.loss import unet_loss
from src.datasets import JinnanDataset
from src.val import valid_aug
from src.unet_plus import *
from src.train_classifer import train_classifer

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
          cfg,
          model,
          optimizer,
          scheduler,
          criterion,
          epochs,
          out_dir,
          trainloader,
          evalloader=None,
          logger=None):
    step = 0
    best_score = 0

    for epoch in range(epochs):
        total_loss = 0
        train_iterator = tqdm.tqdm(trainloader,ncols=50)
        for train_batch, (img, mask) in enumerate(train_iterator):
            optimizer.zero_grad()
            mask = mask.to(device)
            img = img.to(device)
            pred = model(img)
            loss,loss_info = criterion(pred, mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            status = '[{0}] lr = {1:.7f} batch_loss = {2:.3f} epoch_loss = {3:.3f} '.format(
                epoch + 1, scheduler.get_lr()[0], loss.data, total_loss / (train_batch + 1))
            train_iterator.set_description(status)
            if step % 10 == 0:
                for tag, value in loss_info.items():
                    logger.add_scalar(tag, value, step)
            step += 1
        scheduler.step(epoch)
        model_path = os.path.join(out_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        if True:#epoch>20:
            print("Evaluate~~~~~")
            model.eval()
            eval_info = valid_aug(model,evalloader,cfg['img_size'])
            eval_score = eval_info['ious']
            logger.add_scalar('ious', eval_score, step)
            logger.add_scalar('acc', eval_info['acc'], step)
            logger.add_scalar('recall', eval_info['recall'], step)
            model.train()
            if best_score < eval_score:
                best_score = eval_score
                model_path = os.path.join(out_dir, 'best.pth')
                torch.save(model.state_dict(), model_path)
            print("mean ap : {:.4f} , best ap: {:.4f}".format(eval_score, best_score))
            logger.print_info(epoch)


def train_model(model,
                optimizer,
                scheduler,
                cfg):
    for kf in cfg['folds']:
        model = model.to(device)

        trainsets = JinnanDataset(cfg['train_data_dir'],
                                  'kfold_10/restricted_trn_fold_{}.json'.format(kf),
                                  train = True,
                                  img_size = cfg['img_size'])
        trainloader = torch.utils.data.DataLoader(trainsets,
                                                  num_workers=4,
                                                  batch_size=cfg['batch_size'],
                                                  shuffle=True)

        evalsets = JinnanDataset(cfg['train_data_dir'],
                                 'kfold_10/restricted_val_fold_{}.json'.format(kf),
                                 train = False,
                                 img_size = cfg['img_size'])
        evalloader = torch.utils.data.DataLoader(evalsets,
                                                 num_workers=2,
                                                 batch_size=1,
                                                 shuffle=False)

        out_dir = '{}_{}_fold_{}'.format(cfg['model_name'],cfg['img_size'],kf)

        criterion = unet_loss()
        create_dir(out_dir)

        if cfg['finetune_model'] is not None:
             model.load_state_dict(torch.load(cfg['finetune_model']),strict=False)

        logger = txt_logger(out_dir, 'training', 'log.txt')

        train(cfg,model,optimizer,scheduler,
              criterion,cfg['epoches'],out_dir,
              trainloader,evalloader,logger=logger)

        model.load_state_dict(torch.load(os.path.join(out_dir,'best.pth')))

        train_classifer(
            model,
            cfg['classifer_epoch'],
            cfg['train_data_dir'],
            out_dir,
            cfg['img_size']
        )


