##  在fold2 256 base的基础上finetune
cfg = {}
cfg['train_data_dir'] = '/mnt/jinnan2_round2_train_20190401'
cfg['folds'] = [8]
cfg['milestone'] = [6, 12, 18, 30, 40]

cfg['gamma'] = 0.5
cfg['epoches'] = 55
cfg['classifer_epoch'] = 30 

cfg['base_lr'] = 2*1e-4
cfg['batch_size'] = 2 
cfg['img_size'] = 768
cfg['finetune_model'] = None

cfg['model_name'] = 'unet_senet101'
