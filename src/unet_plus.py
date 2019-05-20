from torch import nn
from torch.nn import functional as F
import torch
from src.senet  import  se_resnext50_32x4d,se_resnext101_32x4d

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

class ConvReluBn(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DecoderBlockBN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockBN, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            ConvReluBn(in_channels, middle_channels),
            ConvReluBn(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)



class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2
        b,c,h,w = x.shape
        x_glob = F.upsample(x_glob, size=(w,h), mode='bilinear', align_corners=True)  # 256, 16, 16
        x = x + x_glob

        return x

class SE_Res50UNet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, cls_only = False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = se_resnext50_32x4d()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = self.encoder.layer0
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.cls_only = cls_only


        self.classifer = nn.Sequential(
            se_resnext50_32x4d().layer4,
            nn.Conv2d(2048, 2, 1, stride=1, padding=1, bias=True),
            GlobalAvgPool2d()
        )

        self.center = DecoderBlockBN(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockBN(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockBN(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockBN(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockBN(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockBN(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvReluBn(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.drop = nn.Dropout2d(0.3)

    def load_pred_model(self,model, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model



    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        cls_out = self.classifer(conv4)
        if self.cls_only:
            return cls_out

        conv5 = self.conv5(conv4)
        center = self.center(self.pool(conv5))
        center = self.drop(center)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        pred_mask = self.final(dec0).sigmoid()


        return pred_mask



class SE_Res101UNet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32,
                 pretrained=True, is_deconv=False,cls_only=False):

        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = se_resnext101_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = self.encoder.layer0
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center = FPAv2(2048, num_filters * 8 )#

        self.dec5 = DecoderBlockBN(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockBN(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockBN(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockBN(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockBN(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvReluBn(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.drop = nn.Dropout2d(0.3)

        self.cls_only = cls_only

        self.classifer = nn.Sequential(

            se_resnext101_32x4d().layer4,
            nn.Conv2d(2048, 2, 1, stride=1, padding=1, bias=True),
            GlobalAvgPool2d()
        )

    def load_pred_model(self, model, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        cls_out = self.classifer(conv4)
        if self.cls_only:
            return cls_out


        conv5 = self.conv5(conv4)
        center = self.center(conv5)
        center = self.drop(center)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        pred_mask = self.final(dec0).sigmoid()

        return pred_mask





