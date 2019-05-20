##  在fold2 256 base的基础上finetune
cfg = {}
cfg['train_data_dir'] = '/mnt/jinnan2_round2_train_20190401'
cfg['folds'] = [8]

cfg['epoches'] = 100
cfg['classifer_epoch'] = 30

cfg['base_lr'] = 1e-3
cfg['batch_size'] = 2 
cfg['img_size'] = 960
cfg['finetune_model'] = None

cfg['model_name'] = 'unet_senet50'

