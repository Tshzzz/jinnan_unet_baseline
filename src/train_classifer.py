import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import cv2
import random
from PIL import Image
import tqdm
import numpy as np
from sklearn.model_selection import KFold
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from src.unet_plus import *

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('error to make dir')


def split_train_eval(restrict_path,normal_path):
    random.seed(666)
    path = []
    for p in restrict_path:
        name = os.listdir(p)
        for nn in name:
            path.append([os.path.join(p, nn), 0])
    for p in normal_path:
        name = os.listdir(p)
        for nn in name:
            path.append([os.path.join(p, nn), 1])

    fold = KFold(n_splits=5, shuffle=True, random_state=666)
    n = len(path)
    train = []
    eval = []
    for k, (trn_idx, val_idx) in enumerate(fold.split(list(range(0, n)))):
        train.append([path[i] for i in trn_idx])
        eval.append([path[i] for i in val_idx])
    return train,eval


def img_aug(img):

    if random.random() < 0.5:
        translater = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                cval=(255)
                                ).to_deterministic()
        img = translater.augment_image(img)

    #vertically flip
    if random.random() < 0.5:
        img = np.flip(img,0)

    # horizontally flip
    if random.random() < 0.5:
        img = np.flip(img,1)

    if random.random() < 0.5:
        rot_time = random.choice([1, 2, 3])
        img = np.rot90(img,rot_time)

    return img

class build_dataloader(Dataset):
    def __init__(self,path,train,size=960):
        self.path = path
        self.img_size = size
        self.train = train
    def __len__(self):
        return len(self.path)
    def __getitem__(self, idx):
        img = cv2.imread(self.path[idx][0])
        if self.train:
            img = img_aug(img)
        img = cv2.resize(img,(self.img_size,self.img_size))
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).float()
        id = self.path[idx][1]
        return img/255,id



def train_classifer(
        model,
        epochs,
        imgpath,
        out_dir,
        img_size
):
    model.cls_only = True
    model.eval()
    restricted = [os.path.join(imgpath,'restricted')]
    normal = [os.path.join(imgpath,'normal')]
    train_list ,eval_list = split_train_eval(restricted,normal)
    acc_ = []
    for k in range(0,len(train_list)):
        print('Training Fold {}'.format(k))

        train_datasets = build_dataloader(train_list[k],True,size=img_size)
        trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=35,
                                                  shuffle=True, num_workers=4)

        eval_datasets = build_dataloader(eval_list[k],False,size=img_size)
        evalloader = torch.utils.data.DataLoader(eval_datasets, batch_size=10,
                                                 shuffle=False, num_workers=4)

        create_dir(out_dir)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifer.parameters():
            p.requires_grad = True

        model.classifer.train()
        device = 'cuda'

        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(epochs):
            running_loss = 0.0
            running_corrects = 0
            for imgs,labels in tqdm.tqdm(trainloader,ncols=50):
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step(epoch)
            epoch_loss = running_loss / len(train_datasets)
            epoch_acc = running_corrects.double() / len(train_datasets)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

            eval_corrects = 0
            for imgs,labels in evalloader:
                model.classifer.eval()
                with torch.no_grad():
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    preds = model(imgs)
                    _, preds = torch.max(preds, 1)
                    eval_corrects += torch.sum(preds == labels.data)
                model.classifer.train()
            eval_acc = eval_corrects.data.cpu().double().numpy() / len(eval_datasets)
            if eval_acc >= best_acc:
                best_acc = eval_acc
                model_path = os.path.join(out_dir, 'best.pth')
                torch.save(model.state_dict(), model_path)
            print('{} best: {:.4f} Acc: {:.4f}'.format('Eval', best_acc, eval_acc))
            model_path = os.path.join(out_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
        acc_.append(best_acc)
    print(acc_)
    print(np.array(acc_).mean())
    model.cls_only = False



if __name__ == "__main__":
    #train_classifer()
    pass

