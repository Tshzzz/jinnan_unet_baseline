from src.train import train_model
from config.se101_768_config import cfg
from src.unet_plus import *
import torch.optim.lr_scheduler as lr_scheduler
from src.nadam import Nadam


model = SE_Res101UNet(6)
optimizer = Nadam(model.parameters(), lr=cfg['base_lr'])
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     cfg['milestone'],
                                     gamma=cfg['gamma'])
train_model(model,optimizer,scheduler,cfg)




