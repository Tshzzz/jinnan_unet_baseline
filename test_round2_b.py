import  torch
import numpy as np
import cv2
import  tqdm
import  os
import  json
from pycocotools.mask import *
from src.unet_plus import SE_Res50UNet,SE_Res101UNet
import time

local_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))


TEST_IMG_PATH = '/mnt/jinnan2_round2_test_b_20190424'
NORMAL_LIST_PATH = 'cvfly_normal_list_b.txt'
SUBMIT_PATH   = './submit/cvfly_test_b_{}.json'.format(local_time)


SE50_MODEL_PATH  = './models/se50/best_fold3_se50.pth'
SE101_MODEL_PATH = './models/se101/best_se101.pth'

def get_models(is_clc = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model50 = SE_Res50UNet(6, cls_only=is_clc)
    model50.load_state_dict(torch.load(SE50_MODEL_PATH), strict=True)
    model50 = model50.to(device)
    model50.eval()

    model101 = SE_Res101UNet(6,cls_only = is_clc)
    model101.load_state_dict(torch.load(SE101_MODEL_PATH), strict=True)
    model101 = model101.to(device)
    model101.eval()

    return model50, model101


def clc_aug(img):
    img_list = []
    img_list.append(img.copy())
    img_list.append(np.flipud(img).copy())
    img_list.append(np.fliplr(img).copy())
    return img_list



def clc_aug_tensor(img,size = None):
    img = cv2.resize(img, size)

    assert img.shape[0] == img.shape[1]
    img_list = []
    img_list.append(img.copy())
    img_list.append(np.flipud(img).copy())
    img_list.append(np.fliplr(img).copy())

    img_array = np.array(img_list)
    img_array = torch.from_numpy(img_array).float().permute(0,3,1,2) / 255.
    return img_array



def filter_img_tta(img50,
                    img101,
                   model50,
                   model101,
):
    with torch.no_grad():
        pred50 = model50(img50)
        pred101 = model101(img101)

        pred = pred50 + pred101
        pred = torch.nn.functional.softmax(pred.float(), dim=-1)[0]
        prob = pred[0].data.cpu().numpy()

    return prob > 0.5


def seg_aug_image(img):
    img_list = []
    img_list.append(img)
    img_list.append(np.flipud(img).copy())
    img_list.append(np.fliplr(img).copy())

    img_list.append(np.rot90(img, 1).copy())
    img_list.append(np.rot90(img, 2).copy())
    img_list.append(np.rot90(img, 3).copy())

    return img_list

def seg_restore_mask(img_list):
    img_list[0] = img_list[0]
    img_list[1] = np.flipud(img_list[1])
    img_list[2] = np.fliplr(img_list[2])

    img_list[3] = np.rot90(img_list[3], 3)
    img_list[4] = np.rot90(img_list[4], 2)
    img_list[5] = np.rot90(img_list[5], 1)

    return img_list


def seg_decode_mask(mask_list):
    mask = mask_list[0]
    for i in range(1, len(mask_list)):
        mask += mask_list[i]
    mask = mask/len(mask_list)
    return mask



def seg_aug_image_tensor(img,img_size):
    img = cv2.resize(img, img_size)
    img_list = []
    img_list.append(img)
    img_list.append(np.flipud(img).copy())
    img_list.append(np.fliplr(img).copy())

    img_list.append(np.rot90(img, 1).copy())
    img_list.append(np.rot90(img, 2).copy())
    img_list.append(np.rot90(img, 3).copy())

    img_array = np.array(img_list)
    img_array = torch.from_numpy(img_array).float().permute(0,3,1,2) / 255.
    return img_array



def seg_aug(img_list, model):

    mask_list = []
    with torch.no_grad():
      for i in range(img_list.shape[0]):
        one_img = img_list[i]
        one_img = one_img.unsqueeze(0).cuda()
        one_img = one_img.cuda()
        pred = model(one_img)
        pred = pred[0]
        preds_np = pred.data.cpu().permute(1,2,0).numpy()
        mask_list.append(preds_np)
    mask_list = seg_restore_mask(mask_list)
    mask = seg_decode_mask(mask_list)
    return mask


def make_submit(image_name,preds):
    '''
    Convert the prediction of each image to the required submit format
    :param image_name: image file name
    :param preds: 5 class prediction mask in numpy array
    :return:
    '''

    submit=dict()
    submit['image_name']= image_name
    submit['size']=(preds.shape[1],preds.shape[2])  #(height,width)
    submit['mask']=dict()

    for cls_id in range(0,5):      # 5 classes in this competition

        mask=preds[cls_id,:,:]
        cls_id_str=str(cls_id+1)   # class index from 1 to 5,convert to str
        fortran_mask = np.asfortranarray(mask)
        rle = encode(fortran_mask) #encode the mask into rle, for detail see: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        submit['mask'][cls_id_str]=rle

    return submit



def dump_2_json(submits,save_p):
    '''

    :param submits: submits dict
    :param save_p: json dst save path
    :return:
    '''
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    file = open(save_p, 'w', encoding='utf-8');
    file.write(json.dumps(submits, cls=MyEncoder, indent=4))
    file.close()


from torch.utils.data import Dataset
class cls_tta_dataset(Dataset):
    def __init__(self,path,size50=(960,960),size101=(768,768)):
        self.size50 = size50
        self.size101 = size101
        self.path = path
        self.img_list = os.listdir(path)
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img_path = os.path.join(self.path, name)
        img = cv2.imread(img_path)
        tensor_img50 = clc_aug_tensor(img,self.size50)
        tensor_img101 = clc_aug_tensor(img, self.size101)
        return tensor_img50,tensor_img101,name

#### 分类
def clc():
    model50, model101 = get_models(is_clc=True)
    normal_list = []
    img_list = os.listdir(TEST_IMG_PATH)

    f = open(NORMAL_LIST_PATH, 'w')
    cls_tta = cls_tta_dataset(path=TEST_IMG_PATH,size50=(960,960),size101=(768,768))
    loader = torch.utils.data.DataLoader(cls_tta, batch_size=1, shuffle=False, num_workers=4)

    for img50,img101,name in tqdm.tqdm(loader,ncols=50):
        name = name[0]
        img50 = img50.squeeze(0).cuda()
        img101 = img101.squeeze(0).cuda()
        if not filter_img_tta(img50,img101,
                              model50,
                              model101):
            normal_list.append(name)
            f.write(name + "\n")
    f.close()
    print('normal images: ',len(normal_list))
    print('abnormal images: ',len(img_list) - len(normal_list))


#### 分割
class seg_tta_dataset(Dataset):
    def __init__(self,path,size50=(960,960),size101=(768,768)):
        self.size50 = size50
        self.size101 = size101
        self.path = path
        self.img_list = os.listdir(path)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img_path = os.path.join(self.path, name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        tensor50 = seg_aug_image_tensor(img,self.size50)
        tensor101 = seg_aug_image_tensor(img,self.size101)

        return tensor50,tensor101,name,(h,w)



def seg():
    model50, model101 = get_models(is_clc=False)

    img_list = os.listdir(TEST_IMG_PATH)
    with open(NORMAL_LIST_PATH) as f:
        normal_list = [l.strip() for l in f.readlines()]
    print('normal_list len: ', len(normal_list))
    submits_dict = dict()

    cls_tta = seg_tta_dataset(path=TEST_IMG_PATH,size50=(960,960),size101=(768,768))
    loader = torch.utils.data.DataLoader(cls_tta, batch_size=1, shuffle=False, num_workers=4)

    for tensor50,tensor101,image_id,org_size in tqdm.tqdm(loader,ncols=50):
        h,w = org_size
        image_id = image_id[0]

        if image_id in normal_list:
            preds_np = np.zeros((5, h, w)).astype(np.uint8)
            submit = make_submit(image_id, preds_np)
            submits_dict[image_id] = submit
            continue
        pred50  = seg_aug(tensor50[0], model50)
        pred101 = seg_aug(tensor101[0], model101)

        pred101 = cv2.resize(pred101, (960, 960), interpolation=cv2.INTER_CUBIC)
        pred = (pred50 + pred101) / 2
        pred = np.where(pred > 0.5, 1, 0).astype(np.uint8)
        preds_np = pred[:, :, 1:]


        preds_np = cv2.resize(preds_np, (w, h))
        preds_np = np.transpose(preds_np, (2, 0, 1))
        submit = make_submit(image_id, preds_np)
        submits_dict[image_id] = submit

    dump_2_json(submits_dict, SUBMIT_PATH)


if __name__ == '__main__':
    clc()
    seg()



