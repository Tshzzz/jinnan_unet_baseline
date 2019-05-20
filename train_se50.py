from src.train import train_model
from config.se50_960_config import cfg
from src.unet_plus import *
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

model = SE_Res50UNet(6)
optimizer = optim.SGD(model.parameters(), lr=cfg['base_lr'], momentum=0.9, weight_decay=1e-3)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg['epoches'] // 9) + 1)

train_model(model,optimizer,scheduler,cfg)

