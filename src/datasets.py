from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import random
import torch
from imgaug import augmenters as iaa



def img_aug(img, mask):
    mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    flipper = iaa.Fliplr(0.5).to_deterministic()
    mask = flipper.augment_image(mask)
    img = flipper.augment_image(img)
    vflipper = iaa.Flipud(0.5).to_deterministic()
    img = vflipper.augment_image(img)
    mask = vflipper.augment_image(mask)
    if random.random() < 0.5:
        rot_time = random.choice([1, 2, 3])
        for i in range(rot_time):
            img = np.rot90(img)
            mask = np.rot90(mask)
    if random.random() < 0.5:
        translater = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                shear=(-8, 8),
                                cval=(255)
                                ).to_deterministic()
        img = translater.augment_image(img)
        mask = translater.augment_image(mask)
    # if random.random() < 0.5:
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    return img, mask


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    return False



class JinnanDataset(Dataset):
    def __init__(self,root ,ann_file=None,train=True,img_size = 512,class_num = 5):
        super(JinnanDataset, self).__init__()
        self.class_num = class_num + 1
        self.img_size = img_size
        self.train = train
        self.test = ann_file is None

        if ann_file is None:
            self.img_path = root
            self.img_infos = os.listdir(root)
        else:
            self.img_path = os.path.join(root,'restricted')
            self.normal_path = os.path.join(root,'normal')
            self.normal_list = os.listdir(self.normal_path)
            self.img_infos = self.load_annotations(ann_file)


    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()

        img_infos = []
        '''
        忽略没有框的 box
        '''
        for i in self.img_ids:
            if i in [457,690,760,814,931,989,1199,1361,1805]:##remove hard example
                continue

            ann_ids = self.coco.getAnnIds(imgIds=i, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                info = self.coco.loadImgs([i])[0]
                info['filename'] = info['file_name']
                img_infos.append(info)
            else:
                print(anno)
        return img_infos

    def _parse_ann_info(self, ann_info):
        gt_masks = []
        gt_labels = []
        for i, ann in enumerate(ann_info):
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks.append(self.coco.annToMask(ann))
        ann = dict(masks=gt_masks, labels=gt_labels)
        return ann

    def _cat_img(self,img,mask):
        if random.random() < 0.5:
            if random.random() < 0.2:
                idx = random.randint(0, len(self.normal_list) - 1)
                normal_ = os.path.join(self.normal_path, self.normal_list[idx])
                padding_img = cv2.imread(normal_)
                h, w, _ = padding_img.shape
                paddin_mask = np.zeros((h,w,self.class_num)).astype(np.uint8)
            else:
                idx = random.randint(0, len(self.img_infos) - 1)
                padding_img = self._get_img(idx)
                paddin_mask = self._get_mask(idx)
                h,w,_ = padding_img.shape


            # 找尺寸相近的 resize
            if img.shape[0] - padding_img.shape[0] < img.shape[1] - padding_img.shape[1]:
                padding_img = cv2.resize(padding_img,(padding_img.shape[1],img.shape[0]))
                paddin_mask= cv2.resize(paddin_mask,(padding_img.shape[1],img.shape[0]))
                img = np.concatenate((img, padding_img), axis=1)
                mask = np.concatenate((mask, paddin_mask), axis=1)
            else:
                padding_img = cv2.resize(padding_img,(img.shape[1],padding_img.shape[0]))
                paddin_mask= cv2.resize(paddin_mask,(img.shape[1],paddin_mask.shape[0]))
                img = np.concatenate((img, padding_img), axis=0)
                mask = np.concatenate((mask, paddin_mask), axis=0)

        return img,mask


    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info)

    def _get_mask(self,idx):
        anno_info = self.get_ann_info(idx)
        h,w = anno_info['masks'][0].shape
        mask = np.zeros((h,w,self.class_num))
        for gt,label in zip(anno_info['masks'],anno_info['labels']):
            mask[:,:,int(label)] += gt
            mask[:,:,0] += gt
        mask = np.where(mask>0,255,0).astype(np.uint8)
        return mask

    def _get_img(self,idx):
        if self.test:
            img_path = os.path.join(self.img_path, self.img_infos[idx])
            img = cv2.imread(img_path)
        else:
            img_info = self.img_infos[idx]
            img_path = os.path.join(self.img_path, img_info['file_name'])
            img = cv2.imread(img_path)
        return img


    def perpare_train_val(self,idx):
        img = self._get_img(idx)
        mask = self._get_mask(idx)
        if self.train:
            img, mask = self._cat_img(img, mask)
            img,mask = img_aug(img,mask)

        img = cv2.resize(img,(self.img_size,self.img_size))
        mask = cv2.resize(mask, (self.img_size,self.img_size))
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        img = img.permute(2, 0, 1).float()
        mask = mask.permute(2, 0, 1).float()
        return img, mask

    def perpare_test(self,idx):
        img = self._get_img(idx)
        img = cv2.resize(img,self.img_size)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).float()
        return img

    def __getitem__(self, idx):
        if self.test:
            img = self.perpare_test(idx)
            fname = self.img_infos[idx]
            return img/255.,fname
        else:
            img,mask = self.perpare_train_val(idx)
            return img/255.,mask/255.

    def __len__(self):
        return len(self.img_infos)