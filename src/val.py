import torch
import tqdm
import numpy as np
import cv2
from src.unet_plus import *
from src.datasets  import JinnanDataset

def aug_image(img):
    img_list = []
    img_list.append(img)
    img_list.append(np.flipud(img).copy())
    img_list.append(np.fliplr(img).copy())
    img_list.append(np.rot90(img, 1).copy())
    img_list.append(np.rot90(img, 2).copy())
    img_list.append(np.rot90(img, 3).copy())
    return img_list

def restore_mask(img_list):
    img_list[0] = img_list[0]
    img_list[1] = np.flipud(img_list[1])
    img_list[2] = np.fliplr(img_list[2])
    img_list[3] = np.rot90(img_list[3], 3)
    img_list[4] = np.rot90(img_list[4], 2)
    img_list[5] = np.rot90(img_list[5], 1)
    return img_list


def decode_mask(mask_list):
    mask = mask_list[0]
    for i in range(1, len(mask_list)):
        mask += mask_list[i]
    mask = mask/len(mask_list)
    return mask


def test_aug(img,  model ,img_size=(512, 512)):
    img = cv2.resize(img, img_size)
    img_list = aug_image(img)
    mask_list = []
    with torch.no_grad():
      for one_img in img_list:
        one_img = torch.from_numpy(one_img).float().permute(2, 0, 1) / 255.
        one_img = one_img.unsqueeze(0).cuda()
        one_img = one_img.cuda()
        pred = model(one_img)
        pred = pred[0]
        preds_np = pred.data.cpu().permute(1,2,0).numpy()
        mask_list.append(preds_np)
    mask_list = restore_mask(mask_list)
    mask = decode_mask(mask_list)
    return mask


def valid_aug(model,evalloadr,img_size=512):
    model.eval()
    valid_data = {'ious': [], 'acc': [], 'recall': []}
    with torch.no_grad():
        for img, mask in tqdm.tqdm(evalloadr,ncols=50):
            img  = img*255.
            img  = img[0].permute(1, 2, 0).numpy().astype(np.uint8)
            pred = test_aug(img,  model, (img_size, img_size))
            pred = np.where(pred > 0.5, 1, 0).astype(np.uint8)
            mask = mask[0] > 0.5
            mask = mask.permute(1, 2, 0).numpy()

            pred = pred[:,:, 1:]
            mask = mask[:,:, 1:]
            n_ii = np.logical_and(pred,mask)
            n_ii = np.sum(n_ii).astype(np.float32)
            t_i =  np.sum(mask).astype(np.float32)
            n_ij = np.sum(pred).astype(np.float32)
            ious = n_ii / (t_i + n_ij - n_ii + 1)
            acc = n_ii / (t_i + 1)
            recall = n_ii / (n_ij + 1)

            valid_data['ious'].append(ious)
            valid_data['acc'].append(acc)
            valid_data['recall'].append(recall)

    valid_data['ious'] = np.array(valid_data['ious']).mean()
    valid_data['acc'] = np.array(valid_data['acc']).mean()
    valid_data['recall'] = np.array(valid_data['recall']).mean()

    return valid_data
def valid_aug_ensemble(model1,
                       model2,
                       evalloadr1,
                       evalloadr2,
                       img_size1=640,
                       img_size2=960):
    model1.eval()
    model2.eval()
    valid_data = {'ious': [], 'acc': [], 'recall': []}
    with torch.no_grad():
        evalloadr2_iter = iter(evalloadr2)

        for img, mask in tqdm.tqdm(evalloadr1):
            img  = img*255.
            img  = img[0].permute(1, 2, 0).numpy().astype(np.uint8)
            pred = test_aug(img,  model1, (img_size1, img_size1))



            img2,mask2 = next(evalloadr2_iter)
            img2 = img2 * 255.
            img2 = img2[0].permute(1, 2, 0).numpy().astype(np.uint8)
            pred2 = test_aug(img2, model2, (img_size2, img_size2))
            pred2 = cv2.resize(pred2,(img_size1,img_size1),interpolation=cv2.INTER_CUBIC)
            pred+=pred2
            pred = pred/2


            pred = np.where(pred > 0.39, 1, 0).astype(np.uint8)
            mask = mask[0] > 0.5
            mask = mask.permute(1, 2, 0).numpy()

            pred = pred[:, :, 1:]
            mask = mask[:, :, 1:]

            n_ii = np.logical_and(pred,mask)
            n_ii = np.sum(n_ii).astype(np.float32)
            t_i =  np.sum(mask).astype(np.float32)
            n_ij = np.sum(pred).astype(np.float32)
            ious = n_ii / (t_i + n_ij - n_ii + 1)
            acc = n_ii / (t_i + 1)
            recall = n_ii / (n_ij + 1)

            valid_data['ious'].append(ious)
            valid_data['acc'].append(acc)
            valid_data['recall'].append(recall)

    valid_data['ious'] = np.array(valid_data['ious']).mean()
    valid_data['acc'] = np.array(valid_data['acc']).mean()
    valid_data['recall'] = np.array(valid_data['recall']).mean()

    return valid_data

def local_val2(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SE_Res101UNet(6)
    model.load_state_dict(
        torch.load('./models/se101/best_se101.pth'))


    model = model.to(device)
    evalsets = JinnanDataset(cfg['train_data_dir'],
                             './kfold_10/restricted_val_fold_8.json',
                             train=False, img_size=768)
    evalloader = torch.utils.data.DataLoader(evalsets,
                                              num_workers=4,
                                              batch_size=1,
                                              shuffle=False)


    model2 = SE_Res50UNet(6)

    model2.load_state_dict(torch.load('./models/se50/best_fold3_se50.pth'))

    model2 = model2.to(device)
    evalsets2 = JinnanDataset(cfg['train_data_dir'],
                             './kfold_10/restricted_val_fold_8.json',
                             train=False, img_size=960)

    evalloader2 = torch.utils.data.DataLoader(evalsets2,
                                             num_workers=4,
                                             batch_size=1,
                                             shuffle=False)
    iou = valid_aug_ensemble(model,
                             model2,
                             evalloader,
                             evalloader2,
                             768,
                             960)

    print(iou)
