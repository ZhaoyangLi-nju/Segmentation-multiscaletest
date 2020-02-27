import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# from .resnet import resnet18
# from .resnet import resnet50
# from .resnet import resnet101
# import torchvision.models as models_tv
import model.resnet as resnet_models
import copy
import functools
import util.utils as util
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.decoder import Decoder_Gen, Decoder_Seg, Decoder_Aux
import numpy as np
import cv2
from model.semantic_models import Content_Model, Content_Model_Fast
from model.GatedSpatialConv import GatedSpatialConv2d
from model.layers import ConvNormReLULayer, ConvLayer
from model.contrast_net import Layer_Wise_Contrastive_Net
import model.backbone.resnet as resnet_backbone
import model.backbone.xception as xception_backbone
from model.aspp import ASPP


batch_norm = SynchronizedBatchNorm2d
resnet_models.BatchNorm = batch_norm


# resnet18_place_path = '/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth'

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def fix_grad(net):
    print(net.__class__.__name__)

    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)


def define_netowrks(cfg, device=None):
    # sync bn or not
    # models.BatchNorm = batch_norm
    task_type = cfg.TASK_TYPE

    if task_type == 'segmentation':

        # if cfg.MULTI_SCALE:
        #     # model = FCN_Conc_Multiscale(cfg, device=device)
        #     pass
        # # elif cfg.MULTI_MODAL:
        # #     # model = FCN_Conc_MultiModalTarget_Conc(cfg, device=device)
        # #     # model = FCN_Conc_MultiModalTarget_Late(cfg, device=device)
        # #     model = FCN_Conc_MultiModalTarget(cfg, device=device)
        # else:
        if cfg.MODEL == 'trans2_seg':
            model = Trans2Seg(cfg, device=device)
        elif cfg.MODEL == 'trans2_seg_maxpool':
            model = Trans2Seg_Maxpool(cfg, device=device)
            # model = DUNet()
        # elif cfg.MODEL == 'FCN_MAXPOOL_FAKE':
        #     model = FCN_Conc_Maxpool_FAKE(cfg, device=device)
        # # if cfg.MODEL == 'FCN_LAT':
        # #     model = FCN_Conc_Lat(cfg, device=device)
        # elif cfg.MODEL == 'UNET':
        #     model = UNet(cfg, device=device)
        # elif cfg.MODEL == 'UNET_256':
        #     model = UNet_Share_256(cfg, device=device)
        # elif cfg.MODEL == 'UNET_128':
        #     model = UNet_Share_128(cfg, device=device)
        # elif cfg.MODEL == 'UNET_64':
        #     model = UNet_Share_64(cfg, device=device)
        # elif cfg.MODEL == 'UNET_LONG':
        #     model = UNet_Long(cfg, device=device)
        elif cfg.MODEL == "PSP":
            model = PSPNet(cfg, device=device)
            # model = PSPNet(cfg, BatchNorm=nn.BatchNorm2d, device=device)

    elif task_type == 'recognition':
        if cfg.MODEL == 'trecg':
            model = TRecgNet_Scene_CLS(cfg, device=device)
        elif cfg.MODEL == 'trecg_compl':
            model = TrecgNet_Compl(cfg, device=device)
        elif cfg.MODEL == 'trecg_maxpool':
            model = TrecgNet_Scene_CLS_Maxpool(cfg, device=device)

    return model


# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     print(net)
#     print('Total number of parameters: %d' % num_params)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)


def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]

def set_criterion(cfg, net, device=None, dims=None):

    if 'CLS' in cfg.LOSS_TYPES or cfg.EVALUATE:
        criterion_cls = util.CrossEntropyLoss(weight=cfg.CLASS_WEIGHTS_TRAIN, ignore_index=cfg.IGNORE_LABEL)
        net.set_cls_criterion(criterion_cls)

    if 'SEMANTIC' in cfg.LOSS_TYPES or 'CONTRAST' in cfg.LOSS_TYPES:
        criterion_content = torch.nn.MSELoss()
        # criterion_content = torch.nn.L1Loss()
        content_model = Content_Model_Fast(cfg, criterion_content, device=device, dims=dims)
        # content_model = Content_Model(cfg, criterion_content, device=device, dims=dims)
        net.set_content_model(content_model)

    if 'PIX2PIX' in cfg.LOSS_TYPES:
        criterion_pix2pix = torch.nn.MSELoss()
        # criterion_pix2pix = torch.nn.L1Loss()
        net.set_pix2pix_criterion(criterion_pix2pix)


def expand_Conv(module, in_channels):
    def expand_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.in_channels = in_channels
            m.out_channels = m.out_channels
            mean_weight = torch.mean(m.weight, dim=1, keepdim=True)
            m.weight.data = mean_weight.repeat(1, in_channels, 1, 1).data

    module.apply(expand_func)


##############################################################################
# Moduels
##############################################################################
class Conc_Up_Residual(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True, upsample=True):
        super(Conc_Up_Residual, self).__init__()

        self.upsample = upsample
        if upsample:
            if dim_in == dim_out:
                kernel_size, padding = 3, 1
            else:
                kernel_size, padding = 1, 0

            self.smooth = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
            kernel_size, padding = 1, 0
        else:
            kernel_size, padding = 3, 1

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = norm(dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_out, dim_out)
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.smooth(x)
            residual = x

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.upsmaple:
            x += residual

        return self.relu(x)

# class UpSample(nn.Module):
#     def __init__(self, dim_in, dim_out, padding=nn.ZeroPad2d, norm=nn.BatchNorm2d):
#         super(UpSample, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0),
#             # norm(dim_out),
#             # nn.ReLU(True),
#             padding(1),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
#             norm(dim_out),
#             nn.ReLU(True),
#             padding(1),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
#             norm(dim_out),
#             nn.ReLU(True)
#         )
#
#     def forward(self, x, y=None):
#         up_x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
#         if y is not None:
#             out = torch.cat([up_x, y], dim=1)
#         else:
#             out = up_x
#         return self.model(out)

class UpSample(nn.Module):
    def __init__(self, dim_in, dim_out, padding=nn.ZeroPad2d, norm=nn.BatchNorm2d):
        super(UpSample, self).__init__()

        if dim_in == dim_out:
            dim_in_1 = dim_in
        else:
            dim_in_1 = dim_in // 2
        self.convs1 = conv_norm_relu(dim_in, dim_in_1, kernel_size=1, padding=0)
        self.convs2 = nn.Sequential(
            nn.Conv2d(dim_in_1 * 2, dim_out, kernel_size=1, stride=1, padding=0),
            # norm(dim_out),
            # nn.ReLU(True),
            padding(1),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
            norm(dim_out),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
            norm(dim_out),
            nn.ReLU(True)
        )

    def forward(self, x, y=None):
        x = self.convs1(x)
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
        if y is not None:
            out = torch.cat([x, y], dim=1)
        else:
            out = x
        return self.convs2(out)


class UpSample_2(nn.Module):
    def __init__(self, dim_in, dim_out, padding=nn.ZeroPad2d, norm=nn.BatchNorm2d):
        super(UpSample_2, self).__init__()

        self.convs1 = conv_norm_relu(dim_in, dim_out, kernel_size=1, padding=0)
        dim_out = dim_out // 2
        self.convs2 = nn.Sequential(
            nn.Conv2d(dim_out * 4, dim_out, kernel_size=1, stride=1, padding=0),
            # norm(dim_out),
            # nn.ReLU(True),
            padding(1),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
            norm(dim_out),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
            norm(dim_out),
            nn.ReLU(True)
        )

    def forward(self, x, y=None):
        x = self.convs1(x)
        up_x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
        if y is not None:
            out = torch.cat([up_x, y], dim=1)
        else:
            out = up_x
        return self.convs2(out)


# class UpSample(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super(UpSample, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
#             nn.InstanceNorm2d(dim_out)
#         )
#         self.relu = nn.ReLU(True)
#
#         self.conv2 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0),
#             nn.InstanceNorm2d(dim_out)
#         )
#
#     def forward(self, x, y=None):
#         up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#         if y is not None:
#             out = torch.cat([up_x, y], dim=1)
#         else:
#             out = up_x
#         return self.relu(self.conv2(self.relu(self.conv1(out))))

class Conc_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, padding_type=nn.ZeroPad2d, norm=nn.BatchNorm2d, conc_feat=True, upsample=True):
        super(Conc_Up_Residual_bottleneck, self).__init__()

        self.upsample = upsample
        if dim_in == dim_out:
            kernel_size, padding = 3, 1
        else:
            kernel_size, padding = 1, 0

        self.smooth = nn.Sequential(
            padding_type(padding),
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1, bias=False),
            padding_type(1),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=0, bias=False),
            norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        else:
            dim_in = dim_out

        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(padding_type(1), conv3x3(dim_med, dim_med, padding=0))
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)

        x = self.smooth(x)
        residual = x

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        if self.upsample:
            x += residual

        return self.relu(x)


class Conc_Up_Slim(nn.Module):

    def __init__(self, dim_main, dim_lat, dim_out, stride=1, padding_type=nn.ZeroPad2d, norm=nn.BatchNorm2d, conc_feat=True, upsample=True):
        super(Conc_Up_Slim, self).__init__()

        self.upsample = upsample
        self.relu = nn.ReLU(inplace=True)
        self.smooth = nn.Sequential(
            # padding_type(1),
            nn.Conv2d(dim_main, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out),
            self.relu
        )


        self.conv1 = nn.Conv2d(dim_out + dim_lat, dim_out, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_out)
        self.conv2 = nn.Sequential(padding_type(1), conv3x3(dim_out, dim_out, padding=0))
        self.norm2 = norm(dim_out)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)

        x = self.smooth(x)
        residual = x

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.upsample:
            x += residual

        return self.relu(x)


class Conc_Up_Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, upsample=None, norm=nn.BatchNorm2d, conc_feat=False):
        super(Conc_Up_Bottleneck, self).__init__()

        self.upsample = upsample
        if self.upsample:
            inplanes = planes

        dim_med = int(inplanes / 2)

        self.conv1 = nn.Conv2d(inplanes, dim_med, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(planes)

    def forward(self, x):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.upsample(x)
            residual = x
            x = self.relu(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Conc_Up_BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, upsample=None, norm=nn.BatchNorm2d, conc_feat=False):
        super(Conc_Up_BasicBlock, self).__init__()

        self.upsample = upsample

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.norm1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm(planes)

    def forward(self, x):

        residual = x

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.upsample(x)
            residual = x

        x = self.conv1(x)
        out = self.norm1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out) + residual

        return out


class Add_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, upsample=True):
        super(Add_Up_Residual_bottleneck, self).__init__()

        self.upsample = upsample
        if upsample:
            if dim_in == dim_out:
                kernel_size, padding = 3, 1
            else:
                kernel_size, padding = 1, 0

            self.smooth = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                norm(dim_out))

        dim_in = dim_out

        dim_med = int(dim_out / 2)

        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.smooth(x)
            residual = x

        if y is not None:
            x = x + y

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        if self.upsample:
            x += residual

        return self.relu(x)


class Conc_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Residual_bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))
        # else:
        #     self.residual_conv = nn.Sequential(
        #         nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2,
        #                   padding=1, bias=False),
        #         norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        dim_med = int(dim_out / 2)

        self.conv1 = nn.Conv2d(dim_out, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        if y is not None:
            x = torch.cat((x, y), 1)
        x = self.conv0(x)
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Add_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Add_Residual_bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))

        dim_in = dim_out
        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):
        x = self.conv0(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = x + y

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


##############################################################################
# Trans2 Net
##############################################################################
class BaseTrans2Net(nn.Module):

    def __init__(self, cfg, device=None):
        super(BaseTrans2Net, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        self.arch = cfg.ARCH

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            self.pretrained = True
        else:
            self.pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = resnet_models.__dict__[self.arch](num_classes=365, deep_base=False)
            checkpoint = torch.load('./initmodel/' + self.arch + '_places365.pth', map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('load model pretrained on place')
        else:
            resnet = resnet_models.__dict__[self.arch](pretrained=self.pretrained, deep_base=False)

        self.resnet = resnet
        # self.maxpool = resnet.maxpool
        self.pool = resnet.maxpool
        # self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        # if self.arch == 'resnet18':
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # elif cfg.TASK_TYPE == 'segmentation':
        #     self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
        #                                 resnet.conv3, resnet.bn3, resnet.relu)

        # self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.layer1, self.layer2, self.layer3, self.layer4 = nn.Sequential(self.pool, resnet.layer1), resnet.layer2, resnet.layer3, resnet.layer4
        # self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # if self.trans:
        #     self.aux_net = nn.Sequential(resnet.layer1, resnet.maxpool)
        #     # self.layer1._modules['0'].conv1.stride = (2, 2)
        #     # if cfg.ARCH == 'resnet18':
        #     #     self.layer1._modules['0'].downsample = resnet.maxpool
        #     # else:
        #     #     self.layer1._modules['0'].downsample._modules['0'].stride = (2, 2)
        #
        #     self.build_upsample_content_layers(dims)

        # else:
        #     self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer_1)

        # self.init_type = 'kaiming'
        # if self.pretrained:
        #     if self.trans:
        #         init_weights(self.up1, self.init_type)
        #         init_weights(self.up2, self.init_type)
        #         init_weights(self.up3, self.init_type)
        #         init_weights(self.up4, self.init_type)
        #         init_weights(self.up5, self.init_type)
        # else:
        #     init_weights(self, self.init_type)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    # def _make_upsample(self, block, planes, blocks, stride=1, norm=nn.BatchNorm2d, conc_feat=False):
    #
    #     upsample = None
    #     if stride != 1:
    #         upsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1,
    #                       padding=0, bias=False),
    #             norm(planes)
    #         )
    #
    #     layers = []
    #
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, self.inplanes, norm=norm))
    #
    #     layers.append(block(self.inplanes, planes, upsample, norm, conc_feat))
    #
    #     self.inplanes = planes
    #
    #     return nn.Sequential(*layers)

    # def _make_upsample(self, block, planes, blocks, stride=1, norm=nn.BatchNorm2d, conc_feat=False):
    #
    #     upsample = None
    #     if stride != 1:
    #         if conc_feat:
    #             inplanes = self.inplanes * 2
    #         else:
    #             inplanes = self.inplanes
    #         upsample = nn.Sequential(
    #             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
    #                       padding=0, bias=False),
    #             norm(planes),
    #             nn.ReLU(True)
    #         )
    #         self.inplanes = planes
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, upsample, norm, conc_feat))
    #
    #     for i in range(1, blocks):
    #         layers.append(block(planes, planes, norm=norm))
    #
    #     self.inplanes = planes
    #
    #     return nn.Sequential(*layers)

    # def _make_agant_layer(self, inplanes, planes):
    #
    #     layers = nn.Sequential(
    #         nn.Conv2d(inplanes, planes, kernel_size=3,
    #                   stride=1, padding=1, bias=False),
    #         nn.InstanceNorm2d(planes),
    #         nn.ReLU(inplace=True)
    #     )
    #     return layers

    def build_upsample_content_layers(self, dims):
        pass

class DUNet(nn.Module):
    def __init__(self, encoder_pretrain=False, model_path='', num_class=21):
        super(DUNet, self).__init__()
        self.encoder = resnet_models.Encoder(pretrain=encoder_pretrain, model_path=model_path)
        self.decoder = resnet_models.Decoder(num_class)
    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        x, x_low = self.encoder(source)
        x = self.decoder(x, x_low)

        return x

class BaseTrans2Net_NoPooling(nn.Module):

    def __init__(self, cfg, device=None):
        super(BaseTrans2Net_NoPooling, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        self.arch = cfg.ARCH

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            self.pretrained = True
        else:
            self.pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = resnet_models.__dict__[self.arch](num_classes=365, deep_base=False)
            checkpoint = torch.load('./initmodel/' + self.arch + '_places365.pth', map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'layer_', 'layer'): v for k, v in state_dict.items()}
            resnet.load_state_dict(state_dict)
            print('content model pretrained using place')
        else:
            resnet = resnet_models.__dict__[self.arch](pretrained=self.pretrained, deep_base=False)

        self.maxpool = resnet.maxpool
        self.in_features = resnet.fc.in_features

        # if self.arch == 'resnet18':
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # else:
        #     self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
        #                                 resnet.conv3, resnet.bn3, resnet.relu)

        # self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer_1, resnet.layer_2, resnet.layer_3, resnet.layer_4
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        # self.layer1, self.layer2, self.layer3, self.layer4 = nn.Sequential(resnet.maxpool, resnet.layer1), resnet.layer2, resnet.layer3, resnet.layer4

        if self.arch == 'resnet18' or self.arch == 'alexnet_regular':
            self.latlayer1 = conv_norm_relu(256, 64, kernel_size=3, stride=1, padding=1)
            self.latlayer2 = conv_norm_relu(128, 64, kernel_size=3, stride=1, padding=1)
            self.latlayer3 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.latlayer1 = conv_norm_relu(1024, 256, kernel_size=3, stride=1, padding=1)
            self.latlayer2 = conv_norm_relu(512, 256, kernel_size=3, stride=1, padding=1)
            self.latlayer3 = conv_norm_relu(256, 256, kernel_size=3, stride=1, padding=1)

        if self.trans:
            self.build_upsample_layers(dims)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else batch_norm

        if self.cfg.WHICH_DIRECTION == 'AtoB':
            padding = nn.ReflectionPad2d
        else:
            padding = nn.ZeroPad2d

        if self.arch == 'resnet18' or self.arch == 'alexnet_regular':

            self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
            self.up1 = Conc_Up_Slim(dim_main=256, dim_lat=256, dim_out=256, padding_type=padding, norm=norm)
            self.up2 = Conc_Up_Slim(dim_main=256, dim_lat=128, dim_out=128, padding_type=padding, norm=norm)
            self.up3 = Conc_Up_Slim(dim_main=128, dim_lat=64, dim_out=64, padding_type=padding, norm=norm)
            # self.up1 = Conc_Up_Residual_bottleneck(dims[4] // 1, dims[4] // 2, padding_type=padding, norm=norm)
            # self.up2 = Conc_Up_Residual_bottleneck(dims[4] // 2, dims[4] // 4, padding_type=padding, norm=norm)
            # self.up3 = Conc_Up_Residual_bottleneck(dims[4] // 4, dims[4] // 8, padding_type=padding, norm=norm)
            # self.up1 = UpSample(dims[4] // 1, dims[4] // 2, padding=padding, norm=norm)
            # self.up2 = UpSample(dims[4] // 2, dims[4] // 4, padding=padding, norm=norm)
            # self.up3 = UpSample(dims[4] // 4, dims[4] // 8, padding=padding, norm=norm)
            self.up_image_aux1 = nn.Sequential(
                conv_norm_relu(256, 64, kernel_size=1, padding=0, norm=norm),
                padding(3),
                nn.Conv2d(64, 3, 7, 1, 0, bias=False),
                nn.Tanh()
            )
            self.up_image_aux2 = nn.Sequential(
                conv_norm_relu(128, 64, kernel_size=1, padding=0, norm=norm),
                padding(3),
                nn.Conv2d(64, 3, 7, 1, 0, bias=False),
                nn.Tanh()
            )
            self.up_image_aux3 = nn.Sequential(
                padding(3),
                nn.Conv2d(64, 3, 7, 1, 0, bias=False),
                nn.Tanh()
            )

            # self.latlayer1 = conv_norm_relu(256, 256, kernel_size=3, stride=1, padding=1, norm=norm)
            # self.latlayer2 = conv_norm_relu(128, 128, kernel_size=3, stride=1, padding=1, norm=norm)
            # self.latlayer3 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1, norm=norm)
            # self.smooth1 = conv_norm_relu(384, 128, kernel_size=3, stride=1, padding=1, norm=norm)
            # self.smooth2 = conv_norm_relu(192, 64, kernel_size=3, stride=1, padding=1, norm=norm)
            # self.smooth3 = conv_norm_relu(128, 64, kernel_size=3, stride=1, padding=1, norm=norm)

            if self.cfg.MULTI_SCALE:
                # dim_up_img = 64
                dim_up_img = 256
            else:
                dim_up_img = 448

        elif self.arch == 'resnet50':
            # self.up1 = UpSample(dims[6] // 1, dims[6] // 2, padding=padding, norm=norm)
            # self.up2 = UpSample(dims[6] // 2, dims[6] // 4, padding=padding, norm=norm)
            # self.up3 = UpSample(dims[6] // 4, dims[6] // 16, padding=padding, norm=norm)
            # self.up4 = UpSample_2(dims[6] // 8, dims[6] // 16, padding=padding, norm=norm)
            # self.up1 = Conc_Up_Residual_bottleneck(dims[6] // 1, dims[6] // 2, norm=norm)
            # self.up2 = Conc_Up_Residual_bottleneck(dims[6] // 2, dims[6] // 4, norm=norm)
            # self.up3 = Conc_Up_Residual_bottleneck(dims[6] // 4, dims[6] // 16, norm=norm)
            # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm, conc_feat=False) 128 256 512 1024 2048

            # self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
            # self.up1 = Conc_Up_Slim(dims[6] // 2 + dims[6] // 8, dims[4] // 8, padding_type=padding, norm=norm)
            # self.up2 = Conc_Up_Slim(dims[6] // 4 + dims[6] // 8, dims[4] // 8, padding_type=padding, norm=norm)
            # self.up3 = Conc_Up_Slim(dims[6] // 8 + dims[6] // 8, dims[4] // 8, padding_type=padding, norm=norm)

            self.toplayer = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
            self.up1 = Conc_Up_Slim(dim_main=512, dim_lat=256, dim_out=256, padding_type=padding, norm=norm)
            self.up2 = Conc_Up_Slim(dim_main=256, dim_lat=256, dim_out=256, padding_type=padding, norm=norm)
            self.up3 = Conc_Up_Slim(dim_main=256, dim_lat=256, dim_out=256, padding_type=padding, norm=norm)
            self.up_image_aux1 = nn.Sequential(
                conv_norm_relu(256, 64, kernel_size=1, padding=0, norm=norm),
                padding(3),
                nn.Conv2d(64, 3, 7, 1, 0, bias=False),
                nn.Tanh()
            )
            self.up_image_aux2 = nn.Sequential(
                conv_norm_relu(256, 64, kernel_size=1, padding=0, norm=norm),
                padding(3),
                nn.Conv2d(64, 3, 7, 1, 0, bias=False),
                nn.Tanh()
            )
            self.up_image_aux3 = nn.Sequential(
                conv_norm_relu(256, 64, kernel_size=1, padding=0, norm=norm),
                padding(3),
                nn.Conv2d(64, 3, 7, 1, 0, bias=False),
                nn.Tanh()
            )

            if self.cfg.MULTI_SCALE:
                dim_up_img = 512
            else:
                dim_up_img = 1792

        self.up_image = nn.Sequential(
            nn.Conv2d(dim_up_img, 64, 1, 1, 0, bias=False),
            padding(3),
            nn.Conv2d(64, 3, 7, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        pass


class Trans2Seg_Maxpool(nn.Module):

    def __init__(self, cfg, device=None):
        super(Trans2Seg_Maxpool, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.NUM_CLASSES
        self.device = device
        self.trans = not cfg.NO_TRANS
        self.norm = nn.BatchNorm2d
        self.relu = nn.ReLU()
        # self.norm = SynchronizedBatchNorm2d

        if 'resnet' == cfg.ARCH:
            self.model = resnet_backbone.resnet101(output_stride=32, BatchNorm=self.norm)
            low_feat_dim = 256
        elif 'xception' == cfg.ARCH:
            self.model = xception_backbone.xception()
            low_feat_dim = 128

        self.aspp = ASPP(backbone=cfg.ARCH, output_stride=16, BatchNorm=self.norm, init_type='normal')
        self.decoder_seg = Decoder_Seg(self.cfg, self.norm, low_feat_dim=low_feat_dim, num_classes=self.num_classes, init_type='normal')

        content_dims = []
        if self.trans:
            # self.decoder_gen = Decoder_Gen(nn.InstanceNorm2d, self.num_classes, init_type='normal')
            # self.decoder_aux = Decoder_Aux(self.norm, self.num_classes)

            if cfg.WHICH_CONTENT_NET == 'resnet18':
                content_dims = [64, 64, 128, 256, 512]
            else:
                content_dims = [64, 256, 512, 1024, 2048]

            content_layers = cfg.CONTENT_LAYERS + 1
            dims_content = [content_dims[i] for i in range(content_layers)]

            # content_layers = cfg.CONTENT_LAYERS.split(',')
            # dims_content = [content_dims[int(i)] for i in content_layers]

            self.contrastive_net = Layer_Wise_Contrastive_Net(dims=dims_content)
            self.comparison_net = Comparison_Net()

        # init_type = 'normal'
        # for n, m in self.named_modules():
        #     if 'aux' in n:
        #         init_weights(m, init_type)

        print(self)
        set_criterion(cfg, self, device=device, dims=content_dims)

    def get_1x_lr_params(self):
        modules = [self.model]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder_seg]
        if self.trans:
            modules.append(self.contrastive_net)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def define_aux_net(self, dim_in, reduct=True):

        if reduct:
            dim_out = int(dim_in / 4)
        else:
            dim_out = dim_in
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.2),
            nn.Conv2d(dim_out, self.num_classes, kernel_size=1)
        )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True,segSize=None):
        result = {}
        _, _, H, W = source.size()

        x, low_level_feat2x, low_level_feat4x, low_level_feat8x, low_level_feat16x_1 = self.model(source)
        aspp = self.aspp(x)  # 1/16

        # x, low_level_features = self.model(source)
        # x1 = self.aspp1(x)
        # x2 = self.aspp2(x)
        # x3 = self.aspp3(x)
        # x4 = self.aspp4(x)
        # x5 = self.global_avg_pool(x)
        # x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        #
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        #
        # x = self.concat_projection_conv1(x)
        # x = self.concat_projection_bn1(x)
        # x = self.relu(x)
        # aspp = x

        out_content = None
        # result_seg = self.decoder_seg(x=aspp, low4x=low_level_features)
        result_seg = self.decoder_seg(x=aspp, low4x=low_level_feat4x, low8x=low_level_feat8x,
                                      low16x=low_level_feat16x_1, low2x=low_level_feat2x)
        result['cls'] = F.interpolate(result_seg['score'], size=(H, W), mode='bilinear', align_corners=True)
        if self.trans:
            result['gen_img'] = F.interpolate(result_seg['gen_img'], size=(H, W), mode='bilinear', align_corners=True)

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
            out_content = self.content_model(result['gen_img'], target, layers=content_layers)
            result['loss_content'] = out_content['content_loss']

        if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
            result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CONTRAST' in self.cfg.LOSS_TYPES and cal_loss:

            if out_content is None:
                out_content = self.content_model(result['gen_img'], target, layers=content_layers)

            feat_gen = out_content['feat_gen']
            feat_target = out_content['feat_target']
            feat_target_neg = {k: torch.cat((f[1:], f[0].unsqueeze(0)), dim=0) for k, f in feat_target.items()}

            pos = [torch.cat([gen, feat_target[k]], 1) for k, gen in feat_gen.items()]
            neg = [torch.cat([gen, feat_target_neg[k]], 1) for k, gen in feat_gen.items()]
            result['feat_gen'] = feat_gen
            result['feat_target'] = feat_target

            out = self.contrastive_net(pos, neg)
            result['loss_contrast'] = out['loss_contrast']

            # out_comparison = self.comparison_net(result['gen_img'], target)
            # result['loss_contrast'] = out_comparison['loss_contrast']


        if cal_loss:

            # if out_content is not None:
            #     gen_feat = out_content['feat_gen']['4']
            #     # feat = out_content['feat_gen']['4']
            #     score_semanic = self.semantic_seg(gen_feat)
            #     score_semanic = F.interpolate(score_semanic, size=(H, W), mode='bilinear', align_corners=True)
            #     result['loss_cls'] = self.cls_criterion(result['cls'], label) + self.cls_criterion(score_semanic, label) * 0.3
            # else:
            #     result['loss_cls'] = self.cls_criterion(result['cls'], label)
            result['loss_cls'] = self.cls_criterion(result['cls'], label)
        if segSize!=None:# new add
            result['cls'] = F.interpolate(result['cls'], segSize, mode='bilinear', align_corners=True)
            result['cls'] = nn.functional.softmax(result['cls'], dim=1)
        return result
            #
            # feat_gen = layer_3
            # feat_target = out_content['feat_target']['3']
            # feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)
            #
            # pos = torch.cat((feat_gen, feat_target), 1)
            # neg = torch.cat((feat_gen, feat_target_neg), dim=1)
            #
            # Ej = F.softplus(-pos).mean()
            # Em = F.softplus(neg).mean()
            # result['loss_contrast'] = Em + Ej


            # feat_gen = out_content['feat_gen']['1']
            # feat_target = out_content['feat_target']['1']
            # feat_target = self.edge(feat_target)
            # feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)
            #
            # pos = torch.cat((feat_gen, feat_target), 1)
            # neg = torch.cat((feat_gen, feat_target_neg), dim=1)
            # #
            # Ej = F.softplus(-self.contrastive_net(pos)).mean()
            # Em = F.softplus(self.contrastive_net(neg)).mean()
            # result['loss_contrast'] = Em + Ej


            # feat_gen = self.cross_net(result['gen_img'])
            # feat_target = self.cross_net(target)
            # feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)
            #
            # pos = torch.cat((feat_gen, feat_target), 1)
            # neg = torch.cat((feat_gen, feat_target_neg), dim=1)
            #
            # Ej = F.softplus(-self.contrastive_net(pos)).mean()
            # Em = F.softplus(self.contrastive_net(neg)).mean()
            # result['loss_contrast'] = Em + Ej


            # feat_target_neg_self = {k: torch.cat((f[1:], f[0].unsqueeze(0)), dim=0) for k, f in feat_gen.items()}

            # pos = [torch.cat([gen, feat_target[i]], 1) for i, gen in enumerate(feat_gen)]
            # neg = [torch.cat([gen, feat_target_neg[i]], 1) for i, gen in enumerate(feat_gen)]


            # self_neg = [torch.cat([gen, feat_target_neg_self[k]], 1) for k, gen in feat_gen.items()]

            # loss_contrast = []
            # for i, (pos_f, neg_f) in enumerate(zip(pos, neg)):
            #     Ej = F.softplus(-self.contrastive_net[i](pos_f)).mean()
            #     Em = F.softplus(self.contrastive_net[i](neg_f)).mean()
            #     loss_contrast.append(Em + Ej)
            # result['loss_contrast'] = sum(loss_contrast)

            # contra2
            # com_gen_feat = seg_decode
            # com_target_feat = F.interpolate(gen_decode.detach(), com_gen_feat.size()[2:], mode='bilinear', align_corners=True)
            # # com_target_feat = gen_decode
            # com_target_feat_neg = torch.cat((com_target_feat[1:], com_target_feat[0].unsqueeze(0)), dim=0)
            # com_pos = torch.cat((com_gen_feat, com_target_feat), 1)
            # com_neg = torch.cat((com_gen_feat, com_target_feat_neg), dim=1)
            # Ej = F.softplus(-self.contrastive_net2(com_pos)).mean()
            # Em = F.softplus(self.contrastive_net2(com_neg)).mean()
            # result['loss_contrast'] = Em + Ej

            # result['loss_contrast'] = self.contrastive_net(pos, neg) + (Em + Ej)

            # self_pos = torch.cat((feat_gen, feat_gen), 1)
            # feat_self_neg = torch.cat((feat_gen[1:], feat_gen[0].unsqueeze(0)), dim=0)
            # self_neg = torch.cat((feat_gen, feat_self_neg), dim=1)

            # Ej_self = -F.softplus(-self.d_cross(self_pos)).mean()
            # Em_self = F.softplus(self.d_cross(self_neg)).mean()
            # out['cross_loss_self'] = (Em_self - Ej_self)


                # result['loss_contrast'] = self.contrastive_net(pos, neg, self_neg)


        # result['cls'] = F.interpolate(score, size=(H, W), mode='bilinear',
        #                               align_corners=True) + F.interpolate(score_main, size=(H, W), mode='bilinear',
        #                               align_corners=True)
        # result['cls'] = F.interpolate(score, size=(H, W), mode='bilinear',
        #                               align_corners=True)

        # seg_loss_weight = torch.mean(
        #     F.interpolate(gen_decode2x, size=target.size()[2:], mode='bilinear', align_corners=True), 1,
        #     keepdim=True).repeat(1, 37, 1, 1) / 2 + 0.5

        # atten_feat = gen_decode
        # b, c, h, w = atten_feat.shape
        # feat_att = nn.AvgPool2d(h, 1)(atten_feat).view(b, c, 1, 1)
        # b, c, h, w = gen_decode.shape
        # feat_att = nn.AvgPool2d(h, 1)(atten_decode).view(b, c, 1, 1)

        # atten_decode2 = self.seg_weight(gen_decode2x)
        # b, c, h, w = atten_decode2.shape
        # feat_att2 = nn.AvgPool2d(h, 1)(atten_decode2).view(b, c, 1, 1)

        #score = F.interpolate(self.seg(seg_decode), scale_factor=2, mode='bilinear', align_corners=True)
        # score = F.interpolate(self.seg(seg_decode * feat_att), scale_factor=2, mode='bilinear', align_corners=True)
        # score = score * feat_att2

            # aux = self.aspp2(out_content['feat_gen']['4'])
            # aux = F.interpolate(aux, size=layer_1.size()[2:], mode='bilinear', align_corners=True)
            # aux = torch.cat([aux, out_content['feat_gen']['1']], 1)
            # aux_score = self.score_aux(aux)
            # aux_score = F.interpolate(aux_score, size=(H, W), mode='bilinear',
            #               align_corners=True)
            #
            # result['loss_cls'] = self.cls_criterion(result['cls'], label) + self.cls_criterion(aux_score, label)
            # weight = torch.mean(result['gen_img'], 1, keepdim=True).repeat(1, 37, 1, 1) / 2 + 0.5

            # result['s0'] = torch.mean(result['gen_img'], 1, keepdim=True).repeat(1, 3, 1, 1) / 2 + 0.5
            # result['loss_cls'] = self.cls_criterion(result['cls'], label)

        # if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:


                # result['loss_cls'] = self.cls_criterion(result['cls'], label) + self.cls_criterion(score_aux, label)

                # if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                #     score = self.score_head_semantic(out_content['feat_gen']['4'])
                #     score_aux1 = self.aux1_semantic(out_content['feat_gen']['3'])
                #     score_aux2 = self.aux2_semantic(out_content['feat_gen']['2'])
                #     score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
                #     score = score + score_aux1
                #     score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
                #     score = score + score_aux2
                #     score = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
                #     result['loss_cls'] += self.cls_criterion(score, label)




# class Trans2Seg_Maxpool(BaseTrans2Net):
#
#     def __init__(self, cfg, device=None):
#         super().__init__(cfg, device)
#         self.num_classes = cfg.NUM_CLASSES
#
#         if 'resnet18' == cfg.ARCH:
#             aux_dims = [256, 128, 64, 64]
#             head_dim = 512
#         elif 'resnet50' == cfg.ARCH:
#             aux_dims = [1024, 512, 256, 128, 64]
#             head_dim = 2048
#         elif 'resnet101' == cfg.ARCH:
#             aux_dims = [1024, 512, 256, 128, 64]
#             head_dim = 2048
#
#         if 'resnet18' == self.arch or 'alexnet' in self.arch:
#             self.latlayer1 = conv_norm_relu(256, 64, kernel_size=3, stride=1, padding=1)
#             self.latlayer2 = conv_norm_relu(128, 64, kernel_size=3, stride=1, padding=1)
#             self.latlayer3 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1)
#
#             # self.layer1._modules['0'].conv1.stride = (2, 2)
#             # if cfg.ARCH == 'resnet18':
#             #     self.layer1._modules['0'].downsample = resnet.maxpool
#
#             # for n, m in self.layer3.named_modules():
#             #     if 'conv1' in n:
#             #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#             #     elif 'downsample.0' in n:
#             #         m.stride = (1, 1)
#             # for n, m in self.layer4.named_modules():
#             #     if 'conv1' in n:
#             #         m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
#             #     elif 'downsample.0' in n:
#             #         m.stride = (1, 1)
#
#             self.score_head = ASPP(num_classes=self.num_classes, dim_in=512, norm=batch_norm)
#             self.aux1 = ASPP(num_classes=self.num_classes, dim_in=256, norm=batch_norm)
#             # self.aux2 = ASPP(num_classes=self.num_classes, dim_in=128, norm=batch_norm)
#
#         elif 'resnet50' in self.arch or 'resnet101' in self.arch:
#             self.latlayer1 = conv_norm_relu(1024, 1024, kernel_size=3, stride=1, padding=1)
#             self.latlayer2 = conv_norm_relu(512, 512, kernel_size=3, stride=1, padding=1)
#             self.latlayer3 = conv_norm_relu(256, 256, kernel_size=3, stride=1, padding=1)
#             self.latlayer4 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1)
#
#             # self.layer1._modules['0'].conv1.stride = (2, 2)
#             # self.layer1._modules['0'].downsample._modules['0'].stride = (2, 2)
#
#             # for n, m in self.layer3.named_modules():
#             #     if 'conv2' in n:
#             #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#             #     elif 'downsample.0' in n:
#             #         m.stride = (1, 1)
#             for n, m in self.layer4.named_modules():
#                 if 'conv2' in n:
#                     m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#                     # m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
#                 elif 'downsample.0' in n:
#                     m.stride = (1, 1)
#
#             # self.score_head = nn.Conv2d(head_dim, self.num_classes, kernel_size=1, padding=0)
#             # self.aux1 = nn.Conv2d(aux_dims[0], self.num_classes, kernel_size=1, padding=0)
#             # self.aux2 = nn.Conv2d(aux_dims[1], self.num_classes, kernel_size=1, padding=0)
#             self.aspp = ASPP(norm=batch_norm)
#             # self.aux1 = ASPP(num_classes=self.num_classes, dim_in=1024, norm=batch_norm)
#             # self.aux2 = ASPP(num_classes=self.num_classes, dim_in=512, norm=batch_norm)
#
#             self.u_score = _FCNHead(128, self.num_classes)
#
#
#         # self.score_head = _FCNHead(head_dim, self.num_classes, nn.BatchNorm2d)
#         # self.score_head = nn.Conv2d(head_dim, self.num_classes, kernel_size=1, padding=0)
#
#
#         # if self.trans:
#         #     # self.aux1 = _DenseUpsamplingConvModule(8, aux_dims[1], self.num_classes)
#         #     # self.aux2 = _DenseUpsamplingConvModule(4, aux_dims[2], self.num_classes)
#         #     self.aux1 = nn.Conv2d(aux_dims[1], self.num_classes, kernel_size=1, padding=0)
#         #     self.aux2 = nn.Conv2d(aux_dims[2], self.num_classes, kernel_size=1, padding=0)
#         #
#         #     self.score_head_semantic = nn.Conv2d(512, self.num_classes, kernel_size=1, padding=0)
#         #     self.aux1_semantic = nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0)
#         #     self.aux2_semantic = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
#         # else:
#         #     self.aux1 = nn.Conv2d(aux_dims[0], self.num_classes, kernel_size=1, padding=0)
#         #     self.aux2 = nn.Conv2d(aux_dims[1], self.num_classes, kernel_size=1, padding=0)
#
#         # if self.trans:
#         #     self.aux1 = ASPP(num_classes=self.num_classes, dim_in=256, norm=batch_norm)
#         #     self.aux2 = ASPP(num_classes=self.num_classes, dim_in=256, norm=batch_norm)
#         #     self.score_head_semantic = ASPP(num_classes=self.num_classes, dim_in=head_dim)
#         #     self.aux1_semantic = ASPP(num_classes=self.num_classes, dim_in=2048 // 2)
#         #     self.aux2_semantic = ASPP(num_classes=self.num_classes, dim_in=2048 // 4)
#         # else:
#
#
#         # self.u_score = nn.Conv2d(head_dim, self.num_classes, kernel_size=1, padding=0)
#         # self.duc = _DenseUpsamplingConvModule(32, 512, self.num_classes)
#         # self.aspp_head = ASPP(num_classes=self.num_classes)
#         # self.aspp_1 = ASPP(num_classes=self.num_classes, dim_in=128)
#         # self.aspp_2 = ASPP(num_classes=self.num_classes, dim_in=64, dim_out=32)
#
#         init_type = 'normal'
#         if self.pretrained:
#
#             for n, m in self.named_modules():
#                 if 'up' in n or 'fc' in n or 'score' in n or 'aux' in n or 'cross' in n or 'contrastive' in n:
#                     init_weights(m, init_type)
#         else:
#             init_weights(self, init_type)
#
#         set_criterion(cfg, self)
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def define_aux_net(self, dim_in, reduct=True):
#
#         if reduct:
#             dim_out = int(dim_in / 4)
#         else:
#             dim_out = dim_in
#         return nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1),
#             nn.BatchNorm2d(dim_out),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2),
#             nn.Conv2d(dim_out, self.num_classes, kernel_size=1)
#         )
#
#     # def build_upsample_content_layers(self, dims):
#     #
#     #     norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#     #
#     #     if 'resnet18' == self.cfg.ARCH:
#     #         self.up1 = Conc_Residual_bottleneck(dims[4], dims[3], norm=norm)
#     #         self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
#     #         self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
#     #         self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#     #
#     #     elif 'resnet50' in self.cfg.ARCH or 'resnet101' in self.cfg.ARCH:
#     #
#     #         self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
#     #         self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
#     #         self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#     #         self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
#     #         self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
#     #         # self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
#     #         # self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
#     #         # self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#     #         # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
#     #         # self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
#
#         # self.up_image = nn.Sequential(
#         #     nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#         #     nn.Tanh()
#         # )
#
#     def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
#         result = {}
#         _, _, H, W = source.size()
#
#         layer_0 = self.layer0(source)
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         lat1 = self.latlayer1(layer_3)
#         lat2 = self.latlayer2(layer_2)
#         lat3 = self.latlayer3(layer_1)
#         lat4 = self.latlayer4(layer_0)
#
#         if self.trans:
#             # x = self.tunnel4(layer_4)
#             # x = self.tunnel3(torch.cat([x, layer_3], 1))
#             # x = self.tunnel2(torch.cat([x, layer_2], 1))
#             # up1 = x
#             # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#             # x = self.tunnel1(torch.cat([x, layer_1], 1))
#             # up2 = x
#             # # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#             # # x = self.tunnel0(torch.cat([x, layer_0], 1))
#             # # up4 = x
#             # x = torch.cat([F.interpolate(d, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
#             #                for d in [up1, up2]], dim=1)
#             # x = self.up_image(x)
#             # x = self.tunnel4(layer_4)
#             # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#             # x = self.tunnel3(torch.cat([x, layer_3], 1))
#             # up1 = x
#             # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#             # x = self.tunnel2(torch.cat([x, layer_2], 1))
#             # up2 = x
#             # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#             # x = self.tunnel1(torch.cat([x, layer_1], 1))
#             # up3 = x
#             # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#             # x = self.tunnel0(torch.cat([x, layer_0], 1))
#             # up4 = x
#             # x = torch.cat([F.interpolate(d, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
#             #                for d in [up1, up2, up3, up4]], dim=1)
#             # x = self.up_image(x)
#             # result['gen_img'] = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#
#             # translation branch
#             # up0 = self.toplayer(layer_4)
#             # up = self.up1(layer_4, layer_3)
#             # up1 = up
#             # up = self.up2(up, layer_2)
#             # up2 = up
#             # up = self.up3(up, layer_1)
#             # up3 = up
#             # up = self.up4(up, layer_0)
#             # up4 = up
#             # up = self.up_image(up)
#             # result['gen_img'] = F.interpolate(up, source.size()[2:], mode='bilinear', align_corners=True)
#             # if self.cfg.MULTI_SCALE:
#             #     up = self.up_image(up)
#             # else:
#             #     up = self.up3(up, layer_1)
#             #     up = self.up4(up, layer_0)
#             #     up = self.up_image(up)
#
#             # result['gen_img'] = self.up_image(up)
#             # result['gen_img'] = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#
#             # up = self.toplayer(torch.cat([layer_4, layer_3, layer_2], 1))
#             # up1 = up
#             # up = self.up1(up)
#             # up2 = up
#             # up = self.up3(up, layer_1)
#             # up = self.up_image(up)
#             # result['gen_img'] = F.interpolate(up, size=(H, W), mode='bilinear', align_corners=True)
#
#             _, _, H, W = source.size()
#             # vol = torch.cat([F.interpolate(d, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
#             #                  for d in [up1, up2, up3]], dim=1)
#             # up = self.up_image(vol)
#             # result['gen_img'] = F.interpolate(up, target.size()[2:], mode='bilinear', align_corners=True)
# #
#             up = self.toplayer(layer_4)
#             # up = self.top_res(torch.cat([up, layer_3], 1))
#             up = self.up1(up, lat1)
#             # up = self.up1_res(up)
#             up1 = up
#             up = self.up2(up, lat2)
#             # up = self.up2_res(up)
#             up2 = up
#             up = self.up3(up, lat3)
#             # up = self.up3_res(up)
#             up3 = up
#             up = self.up4(up, lat4)
#             up4 = up
# #
# #             result['gen_img'] = F.interpolate(self.up_image_aux4(up4), size=(H, W), mode='bilinear', align_corners=True)
# #             # img1 = F.interpolate(self.up_image_aux1(up1), size=(H, W), mode='bilinear', align_corners=True)
# #             # img2 = F.interpolate(self.up_image_aux2(up2), size=(H, W), mode='bilinear', align_corners=True)
# #             # img3 = F.interpolate(self.up_image_aux3(up3), size=(H, W), mode='bilinear', align_corners=True)
# #             # img4 = F.interpolate(self.up_image_aux4(up4), size=(H, W), mode='bilinear', align_corners=True)
# #             # result['gen_img'] = F.interpolate(img1 + img2 + img3 + img4, size=(H, W), mode='bilinear', align_corners=True)
# #
# #             if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
# #                 out_content = self.content_model(result['gen_img'], target, layers=content_layers)
# #                 result['loss_content'] = out_content['content_loss']
# #
# #             if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
# #                 result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)
# #
# #         if 'CONTRAST' in self.cfg.LOSS_TYPES and cal_loss:
# #
# #             out_content = self.content_model(target, target, layers=content_layers)
# #
# #             feat_gen = layer_3
# #             feat_target = out_content['feat_target']['3']
# #             feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)
# #
# #             pos = torch.cat((feat_gen, feat_target), 1)
# #             neg = torch.cat((feat_gen, feat_target_neg), dim=1)
# #
# #             Ej = F.softplus(-pos).mean()
# #             Em = F.softplus(neg).mean()
# #             result['loss_contrast'] = Em + Ej
# #                 # feat_gen = self.cross_net(result['gen_img'])
# #                 # feat_target = self.cross_net(target)
# #                 # feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)
# #                 #
# #                 # pos = torch.cat((feat_gen, feat_target), 1)
# #                 # neg = torch.cat((feat_gen, feat_target_neg), dim=1)
# #                 #
# #                 # Ej = F.softplus(-self.contrastive_net(pos)).mean()
# #                 # Em = F.softplus(self.contrastive_net(neg)).mean()
# #                 # result['loss_contrast'] = Em + Ej
# #
# #         if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:
# #
# #             # score = self.duc(layer_4)
# #             # score = self.score_head(layer_4)
# #             #
# #             # if self.trans:
# #             #
# #             #     score_aux1 = self.aux1(up1)
# #             #     score_aux2 = self.aux2(up2)
# #             #
# #             # else:
# #             #     score_aux1 = self.aux1(layer_3)
# #             #     score_aux2 = self.aux2(layer_2)
# #             # #
# #             # #     # if self.cfg.WHICH_SCORE == 'main' or not self.trans:
# #             # #     #     score_aux1 = self.aux1(layer_3)
# #             # #     #     score_aux2 = self.aux2(layer_2)
# #             # #     #
# #             # #     # elif self.cfg.WHICH_SCORE == 'up':
# #             # #     #
# #             # #     #     score_aux1 = self.aux1(up1)
# #             # #     #     score_aux2 = self.aux2(up2)
# #             # #     # elif self.cfg.WHICH_SCORE == 'both':
# #             # #     #
# #             # #     #     score_aux1 = self.aux1(layer_3) + self.aux1_up(up1)
# #             # #     #     score_aux2 = self.aux2(layer_2) + self.aux2_up(up2)
# #             # #     # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
# #             # #     # score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
# #             # #     # score = score + score_aux1
# #             # #     # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
# #             # #     # score = score + score_aux2
# #             # #     # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
# #             # #
# #             # score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
# #             # score = score + score_aux1
# #             # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
# #             # score = score + score_aux2
# #             # # score = self.u_score(vol)
# #             # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
# #
# #             score = self.score_head(layer_4)  # (shape: (batch_size, num_classes, h/16, w/16))
# #             if self.trans:
# #                 score_aux1 = self.aux1(up1)
# #                 score_aux2 = self.aux2(up2)
# #             else:
# #                 score_aux1 = self.aux1(layer_3)
# #                 # score_aux2 = self.aux2(layer_2)
# #
# #             score = F.interpolate(score, size=score_aux1.size()[2:], mode='bilinear', align_corners=True)
# #             score = score + score_aux1
# #             # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
# #             # score = score + score_aux2
# #             result['cls'] = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=True)  # (shape: (batch_size, num_classes, h, w))
# #
# #             # score = self.u_score(up4)
# #             # result['cls'] = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=True)
# #             if cal_loss:
# #                 result['loss_cls'] = self.cls_criterion(result['cls'], label)
# #
# #                 # if 'SEMANTIC' in self.cfg.LOSS_TYPES:
# #                 #     score = self.score_head_semantic(out_content['feat_gen']['4'])
# #                 #     score_aux1 = self.aux1_semantic(out_content['feat_gen']['3'])
# #                 #     score_aux2 = self.aux2_semantic(out_content['feat_gen']['2'])
# #                 #     score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
# #                 #     score = score + score_aux1
# #                 #     score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
# #                 #     score = score + score_aux2
# #                 #     score = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
# #                 #     result['loss_cls'] += self.cls_criterion(score, label)
# #
# #         return result


class _DenseUpsamplingConvModule(nn.Module):
    def __init__(self, down_factor, in_dim, num_classes):
        super(_DenseUpsamplingConvModule, self).__init__()
        upsample_dim = (down_factor ** 2) * num_classes
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(upsample_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class Trans2Seg(BaseTrans2Net_NoPooling):

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.num_classes = cfg.NUM_CLASSES

        if 'resnet18' == cfg.ARCH:
            aux_dims = [256, 128, 64, 64]
            head_dim = 512
        elif 'resnet50' == cfg.ARCH:
            aux_dims = [1024, 512, 256, 128, 64]
            head_dim = 2048
        elif 'resnet101' == cfg.ARCH:
            aux_dims = [1024, 512, 256, 128, 64]
            head_dim = 2048

        # self.score_head = _FCNHead(head_dim, self.num_classes, nn.BatchNorm2d)
        # self.score_head = nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
        self.score_head = ASPP(num_classes=self.num_classes, dim_in=2048, norm=batch_norm)
        # self.score_u = nn.Conv2d(sum([aux_dims[i] for i in range(3)]), self.num_classes, kernel_size=1)
        # if self.trans:
            # self.aux1 = nn.Conv2d(aux_dims[0], self.num_classes, kernel_size=1)
            # self.aux2 = nn.Conv2d(aux_dims[1], self.num_classes, kernel_size=1)
        if self.trans:
            self.aux1 = ASPP(num_classes=self.num_classes, dim_in=256, norm=batch_norm)
            self.aux2 = ASPP(num_classes=self.num_classes, dim_in=256, norm=batch_norm)
        else:
            self.aux1 = ASPP(num_classes=self.num_classes, dim_in=1024, norm=batch_norm)
            self.aux2 = ASPP(num_classes=self.num_classes, dim_in=512, norm=batch_norm)
            # self.aux1 = nn.Conv2d(aux_dims[0] * 2, self.num_classes, kernel_size=1)
            # self.aux2 = nn.Conv2d(aux_dims[1] * 2, self.num_classes, kernel_size=1)
        # else:
        #     self.aux1 = nn.Conv2d(aux_dims[0], self.num_classes, kernel_size=1)
        #     self.aux2 = nn.Conv2d(aux_dims[1], self.num_classes, kernel_size=1)

        # self.score_head_semantic = nn.Conv2d(head_dim + sum([aux_dims[i] for i in range(2)]), self.num_classes, kernel_size=1)
        self.score_head_semantic = nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
        # self.score_head_semantic = _FCNHead(head_dim, self.num_classes, nn.BatchNorm2d)
        # self.aux1_semantic = _FCNHead(aux_dims[0], self.num_classes, nn.BatchNorm2d)
        # self.aux2_semantic = _FCNHead(aux_dims[1], self.num_classes, nn.BatchNorm2d)
        self.aux1_semantic = nn.Conv2d(aux_dims[0], self.num_classes, kernel_size=1)
        self.aux2_semantic = nn.Conv2d(aux_dims[1], self.num_classes, kernel_size=1)
        init_type = 'normal'

        if self.pretrained:

            # if self.trans:
            #     init_weights(self.up1, init_type)
            #     init_weights(self.up2, init_type)
            #     init_weights(self.up3, init_type)
            #     init_weights(self.up4, init_type)
            #     init_weights(self.up5, init_type)
            #     init_weights(self.up_image, init_type)

            for n, m in self.named_modules():
                if 'aux' in n or 'up' in n or 'score' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

        set_criterion(cfg, self)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def define_aux_net(self, dim_in, reduct=True):

        if reduct:
            dim_out = int(dim_in / 4)
        else:
            dim_out = dim_in
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(dim_out, self.num_classes, kernel_size=1)
        )

    # def build_upsample_content_layers(self, dims):
    #
    #     norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
    #
    #     if 'resnet18' == self.cfg.ARCH:
    #         self.up1 = Conc_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
    #         self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
    #
    #     elif 'resnet50' in self.cfg.ARCH or 'resnet101' in self.cfg.ARCH:
    #
    #         self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
    #         self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
    #         self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)
    #         # self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
    #         # self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
    #         # self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
    #         # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
    #         # self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        # self.up_image = nn.Sequential(
        #     nn.Conv2d(64, 3, 7, 1, 3, bias=False),
        #     nn.Tanh()
        # )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True,segSize=None):
        result = {}

        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        lat1 = self.latlayer1(layer_3)
        lat2 = self.latlayer2(layer_2)
        lat3 = self.latlayer3(layer_1)

        if self.trans:
            # # up0 = self.toplayer(layer_4)
            # up = self.up1(layer_4, layer_3)
            # up1 = up
            # up = self.up2(up, layer_2)
            # up2 = up
            # up = self.up3(up, layer_1)
            # up3 = up
            #
            _, _, H, W = source.size()
            # vol = torch.cat([F.interpolate(d, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
            #                  for d in [up1, up2, up3]], dim=1)
            # up = self.up_image(vol)
            # result['gen_img'] = F.interpolate(up, target.size()[2:], mode='bilinear', align_corners=True)

            up = self.toplayer(layer_4)

            up = self.up1(up, lat1)
            up1 = up
            up = self.up2(up, lat2)
            up2 = up
            up = self.up3(up, lat3)
            up3 = up

            img1 = F.interpolate(self.up_image_aux1(up1), size=(H, W), mode='bilinear', align_corners=True)
            img2 = F.interpolate(self.up_image_aux2(up2), size=(H, W), mode='bilinear', align_corners=True)
            img3 = F.interpolate(self.up_image_aux3(up3), size=(H, W), mode='bilinear', align_corners=True)
            result['gen_img'] = F.interpolate(img1 + img2 + img3, size=(H, W), mode='bilinear', align_corners=True)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                out_content = self.content_model(result['gen_img'], target, layers=content_layers)
                result['loss_content'] = out_content['content_loss']

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

        if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:

            # segmentation branch


            # if self.cfg.WHICH_SCORE == 'main' or not self.trans:
            #     score_aux1 = self.aux1(layer_3)
            #     score_aux2 = self.aux2(layer_2)
            #
            # elif self.cfg.WHICH_SCORE == 'up':
            #
            #     score_aux1 = self.aux1(up1)
            #     score_aux2 = self.aux2(up2)
            # elif self.cfg.WHICH_SCORE == 'both':
            #
            #     score_aux1 = self.aux1(layer_3) + self.aux1_up(up1)
            #     score_aux2 = self.aux2(layer_2) + self.aux2_up(up2)
            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
            # score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux1
            # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux2
            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            score = self.score_head(layer_4)
            if self.trans:

                # score_aux1 = self.aux1(torch.cat((layer_3, up1), 1))
                # score_aux2 = self.aux2(torch.cat((layer_2, up2), 1))

                score_aux1 = self.aux1(up1)
                score_aux2 = self.aux2(up2)

            else:
                score_aux1 = self.aux1(layer_3)
                score_aux2 = self.aux2(layer_2)

            score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1
            score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux2
            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            # score = self.score_u(vol)
            # result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)
                # if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                #     score = self.score_head_semantic(out_content['feat_gen']['4'])
                #     score_aux1 = self.aux1_semantic(out_content['feat_gen']['3'])
                #     score_aux2 = self.aux2_semantic(out_content['feat_gen']['2'])
                #     score = F.interpolate(score, score_aux1.size()[2:], mode='bilinear', align_corners=True)
                #     score = score + score_aux1
                #     score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
                #     score = score + score_aux2
                #     score = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
                #     result['loss_cls'] += self.cls_criterion(score, label) * 0.3

                    # feat = torch.cat([F.interpolate(feat, source.size()[2:], mode='bilinear', align_corners=True)
                    #         for feat in [out_content['feat_gen']['4'], out_content['feat_gen']['3'], out_content['feat_gen']['3']]], 1)
                    # score = self.score_head_semantic(feat)
                    # result['loss_cls'] += self.cls_criterion(score, label)
        if segSize!=None:#new add
            result['cls'] = F.interpolate(result['cls'], segSize, mode='bilinear', align_corners=True)
            result['cls'] = nn.functional.softmax(result['cls'], dim=1)
        return result


# class FCN_Conc_Maxpool_FAKE(nn.Module):
#
#     def __init__(self, cfg, device=None):
#         super(FCN_Conc_Maxpool_FAKE, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#
#         self.source_net = FCN_Conc_Maxpool(cfg, device)
#         import copy
#         cfg_sample = copy.deepcopy(cfg)
#         cfg_sample.USE_FAKE_DATA = False
#         cfg_sample.NO_TRANS = True
#         self.compl_net = FCN_Conc_Maxpool(cfg_sample, device)
#
#     def set_content_model(self, content_model):
#         self.source_net.set_content_model(content_model)
#
#     def set_cls_criterion(self, criterion):
#         self.source_net.set_cls_criterion(criterion)
#         self.compl_net.set_cls_criterion(criterion)
#
#     def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
#
#         result_source = self.source_net(source, target, label, phase, content_layers, cal_loss=cal_loss)
#         input_compl = result_source['gen_img'].detach()
#         result_compl = self.compl_net(input_compl, None, label, phase, content_layers, cal_loss=cal_loss)
#
#         if phase == 'train':
#             result_source['loss_cls_compl'] = result_compl['loss_cls']
#         else:
#             result_source['cls'] += result_compl['cls']
#             # result_source['cls_compl'] = result_compl['cls']
#             # result_source['cls_fuse'] = (result_source['cls'] + result_compl['cls']) * 0.5
#
#         return result_source


class TrecgNet_Compl(nn.Module):

    def __init__(self, cfg, device=None):
        super(TrecgNet_Compl, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device

        self.source_net = TRecgNet_Scene_CLS(cfg, device)
        import copy
        cfg_sample = copy.deepcopy(cfg)
        cfg_sample.USE_FAKE_DATA = False
        cfg_sample.NO_TRANS = True
        # cfg_sample.PRETRAINED = ''
        self.compl_net = TrecgNet_Scene_CLS_Maxpool(cfg_sample, device)
        # if cfg.MULTI_SCALE:
        #     for n, m in self.compl_net.layer4.named_modules():
        #         if 'conv1' in n:
        #             m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #         elif 'downsample.0' in n:
        #             m.stride = (1, 1)
        #     self.compl_net.avgpool = nn.AvgPool2d(7, 1)
        # else:
        # self.compl_net.avgpool = nn.AvgPool2d(14, 1)

        set_criterion(cfg, self)
        set_criterion(cfg, self.source_net)
        set_criterion(cfg, self.compl_net)

        # self.fc_compl = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(512, cfg.NUM_CLASSES)
        # )

        # init_weights(self.compl_net, 'normal')
        # init_weights(self.fc_compl, 'normal')
        # self.avgpool = nn.AvgPool2d(14, 1)
        # for n, m in self.compl_net.layer4.named_modules():
        #     if 'conv1' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)

        # self.fc = nn.Sequential(
        #     conv_norm_relu(1024, 512, kernel_size=1, stride=1, padding=0),
        #     conv_norm_relu(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.AvgPool2d(14, 1),
        #     Flatten(),
        #     nn.Linear(512, self.cfg.NUM_CLASSES)
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, cfg.NUM_CLASSES)
        # )

    def set_content_model(self, content_model):
        self.source_net.set_content_model(content_model)

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.source_net.set_cls_criterion(criterion)
        self.compl_net.set_cls_criterion(criterion)
        self.cls_criterion = criterion

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):

        result_source = self.source_net(source, target, label, phase, content_layers, cal_loss=cal_loss)

        # if self.cfg.INFERENCE or phase == 'test':
        #     input_compl = target
        # else:
        #     input_compl = result_source['gen_img'].detach()

        input_compl = result_source['gen_img'].detach()
        result_source['compl_source'] = input_compl
        result_compl = self.compl_net(input_compl, label=label)
        # cls_compl = self.fc_compl(F.avg_pool2d(feat_compl, feat_compl.size()[-1]))
        # result_compl = self.compl_net(input_compl, None, label, phase, content_layers, cal_loss=cal_loss)

        # conc_feat = torch.cat([result_source['avgpool'], result_compl['avgpool']], 1).to(self.device)
        # result_source['cls'] = self.fc(flatten(conc_feat))

        # cls_fuse = self.fc(cat)
        # if phase == 'train':
        #     result_source['loss_cls_compl'] = result_compl['loss_cls']
        #     result_source['loss_cls_fuse'] = self.cls_criterion(cls_fuse, label)
        # result_source['cls'] = cls_fuse
        if phase == 'train' and cal_loss:
            # result_source['loss_cls_compl'] = self.cls_criterion(cls_compl, label)
            result_source['loss_cls_compl'] = result_compl['loss_cls']
            result_source['loss_cls'] = self.cls_criterion(result_source['cls'], label)

        result_source['cls_compl'] = result_compl['cls']
        result_source['cls_original'] = result_source['cls']
        # result_source['avgpool_compl'] = result_compl['avgpool']
        alpha_main = 0.75
        result_source['cls'] = result_source['cls'] * alpha_main + result_source['cls_compl'] * (1-alpha_main)
        # result_source['cls'] = result_source['cls'] * 0.7 + result_source['cls_compl'] * 0.3

        return result_source


# class FCN_Conc_Maxpool(nn.Module):
#
#     def __init__(self, cfg, device=None):
#         super(FCN_Conc_Maxpool, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         resnet = models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=False)
#
#         self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
#         # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
#         #                             resnet.conv3, resnet.bn3, resnet.relu)
#         self.maxpool = resnet.maxpool
#         self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
#
#         if self.trans:
#             if 'resnet50' in self.cfg.ARCH:
#                 for n, m in self.layer3.named_modules():
#                     if 'conv2' in n:
#                         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#                     elif 'downsample.0' in n:
#                         m.stride = (1, 1)
#                 for n, m in self.layer4.named_modules():
#                     if 'conv2' in n:
#                         m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
#                     elif 'downsample.0' in n:
#                         m.stride = (1, 1)
#             # elif 'resnet18' in self.cfg.ARCH:
#             #     for n, m in self.layer4.named_modules():
#             #         if 'conv1' in n:
#             #             m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#             #         elif 'downsample.0' in n:
#             #             m.stride = (1, 1)
#
#             self.build_upsample_content_layers(dims)
#
#         if 'resnet18' == cfg.ARCH:
#             aux_dims = [256, 128, 64]
#             head_dim = 512
#         elif 'resnet50' == cfg.ARCH:
#             aux_dims = [512, 256, 64]
#             head_dim = 2048
#
#         self.score_head = _FCNHead(head_dim, num_classes, batch_norm)
#
#         self.score_aux1 = nn.Sequential(
#             nn.Conv2d(aux_dims[0], num_classes, 1)
#         )
#
#         self.score_aux2 = nn.Sequential(
#             nn.Conv2d(aux_dims[1], num_classes, 1)
#         )
#         self.score_aux3 = nn.Sequential(
#             nn.Conv2d(aux_dims[2], num_classes, 1)
#         )
#
#         init_type = 'normal'
#         if pretrained:
#             init_weights(self.score_head, init_type)
#
#             if self.trans:
#                 init_weights(self.up1, init_type)
#                 init_weights(self.up2, init_type)
#                 init_weights(self.up3, init_type)
#                 init_weights(self.up4, init_type)
#                 init_weights(self.cross_layer_3, init_type)
#                 init_weights(self.cross_layer_4, init_type)
#
#             init_weights(self.score_head, init_type)
#             init_weights(self.score_aux3, init_type)
#             init_weights(self.score_aux2, init_type)
#             init_weights(self.score_aux1, init_type)
#
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def build_upsample_content_layers(self, dims):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         if 'resnet18' == self.cfg.ARCH:
#             self.up1 = Conc_Residual_bottleneck(dims[4], dims[3], norm=norm)
#             self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
#             self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
#             self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         elif 'resnet50' in self.cfg.ARCH:
#             self.cross_layer_4 = nn.Conv2d(dims[6], dims[4], kernel_size=1, bias=False)
#             self.cross_layer_3 = nn.Conv2d(dims[5], dims[4], kernel_size=1, bias=False)
#
#             self.up1 = Conc_Residual_bottleneck(dims[5], dims[4], norm=norm)
#             self.up2 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#             self.up3 = Conc_Up_Residual_bottleneck(dims[3], dims[1], norm=norm)
#             self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         self.up_image = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
#         result = {}
#
#         layer_0 = self.layer0(source)
#         layer_1 = self.layer1(self.maxpool(layer_0))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         if self.trans:
#             # translation branch
#             cross_layer4 = self.cross_layer_4(layer_4)
#             cross_layer3 = self.cross_layer_3(layer_3)
#
#             cross_conc = torch.cat((cross_layer4, cross_layer3), 1)
#
#             up1 = self.up1(cross_conc, layer_2)
#             up2 = self.up2(up1, layer_1)
#             up3 = self.up3(up2, layer_0)
#             up4 = self.up4(up3)
#
#             result['gen_img'] = self.up_image(up4)
#
#             if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
#                 result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
#
#         if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:
#
#             # segmentation branch
#             score_head = self.score_head(layer_4)
#
#             score_aux1 = None
#             score_aux2 = None
#             score_aux3 = None
#             if self.cfg.WHICH_SCORE == 'main' or not self.trans:
#                 score_aux1 = self.score_aux1(layer_3)
#                 score_aux2 = self.score_aux2(layer_2)
#                 score_aux3 = self.score_aux3(layer_1)
#             elif self.cfg.WHICH_SCORE == 'up':
#                 score_aux1 = self.score_aux1(up1)
#                 # score_aux2 = self.score_aux2(up2)
#                 # score_aux3 = self.score_aux3(up3)
#
#             score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux1
#             score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux2
#             score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux3
#
#             result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
#
#             if cal_loss:
#                 result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result


# class FCN_Conc_MultiModalTarget(nn.Module):
#
#     def __init__(self, cfg, device=None):
#         super(FCN_Conc_MultiModalTarget, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = resnet_models.__dict__['resnet18'](num_classes=365)
#             checkpoint = torch.load(resnet18_place_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.score_head = _FCNHead(512, num_classes)
#
#         if self.trans:
#             self.build_upsample_content_layers(dims)
#
#         self.score_aux1 = nn.Sequential(
#             nn.Conv2d(dims[3] * 2, num_classes, 1)
#         )
#
#         self.score_aux2 = nn.Sequential(
#             nn.Conv2d(dims[2] * 2, num_classes, 1)
#         )
#         self.score_aux3 = nn.Sequential(
#             nn.Conv2d(dims[1] * 2, num_classes, 1)
#         )
#
#         init_type = 'normal'
#         if pretrained:
#
#             init_weights(self.score_head, init_type)
#
#             if self.trans:
#                 init_weights(self.up1_depth, init_type)
#                 init_weights(self.up2_depth, init_type)
#                 init_weights(self.up3_depth, init_type)
#                 init_weights(self.up4_depth, init_type)
#                 init_weights(self.up1_seg, init_type)
#                 init_weights(self.up2_seg, init_type)
#                 init_weights(self.up3_seg, init_type)
#                 init_weights(self.up4_seg, init_type)
#
#             init_weights(self.score_aux3, init_type)
#             init_weights(self.score_aux2, init_type)
#             init_weights(self.score_aux1, init_type)
#             init_weights(self.score_head, init_type)
#
#         else:
#
#             init_weights(self, init_type)
#
#     def set_content_model(self, content_model):
#         self.content_model = content_model
#
#     def set_pix2pix_criterion(self, criterion):
#         self.pix2pix_criterion = criterion.to(self.device)
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def build_upsample_content_layers(self, dims):
#
#         norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#
#         if 'bottleneck' in self.cfg.FILTERS:
#             self.up1_depth = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#             self.up2_depth = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
#             self.up3_depth = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
#             self.up4_depth = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#
#             self.up1_seg = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
#             self.up2_seg = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
#             self.up3_seg = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
#             self.up4_seg = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)
#         else:
#             self.up1_depth = Conc_Up_Residual(dims[4], dims[3], norm=norm)
#             self.up2_depth = Conc_Up_Residual(dims[3], dims[2], norm=norm)
#             self.up3_depth = Conc_Up_Residual(dims[2], dims[1], norm=norm)
#             self.up4_depth = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
#
#             self.up1_seg = Conc_Up_Residual(dims[4], dims[3], norm=norm)
#             self.up2_seg = Conc_Up_Residual(dims[3], dims[2], norm=norm)
#             self.up3_seg = Conc_Up_Residual(dims[2], dims[1], norm=norm)
#             self.up4_seg = Conc_Up_Residual(dims[1], dims[1], norm=norm, conc_feat=False)
#
#         self.up_depth = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#         self.up_seg = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, source=None, target_1=None, target_2=None, label=None, phase='train', content_layers=None,
#                 cal_loss=True):
#         result = {}
#         layer_0 = self.relu(self.bn1(self.conv1(source)))
#         layer_1 = self.layer1(layer_0)
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         if self.trans:
#             # content model branch
#             up1_depth = self.up1_depth(layer_4, layer_3)
#             up2_depth = self.up2_depth(up1_depth, layer_2)
#             up3_depth = self.up3_depth(up2_depth, layer_1)
#             up4_depth = self.up4_depth(up3_depth)
#             result['gen_depth'] = self.up_depth(up4_depth)
#
#             up1_seg = self.up1_seg(layer_4, layer_3)
#             up2_seg = self.up2_seg(up1_seg, layer_2)
#             up3_seg = self.up3_seg(up2_seg, layer_1)
#             up4_seg = self.up4_seg(up3_seg)
#             result['gen_seg'] = self.up_seg(up4_seg)
#
#             if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
#                 result['loss_content_depth'] = self.content_model(result['gen_depth'], target_1, layers=content_layers)
#                 result['loss_content_seg'] = self.content_model(result['gen_seg'], target_2, layers=content_layers)
#
#         if 'CLS' in self.cfg.LOSS_TYPES or self.cfg.INFERENCE:
#
#             score_head = self.score_head(layer_4)
#
#             score_aux1 = None
#             score_aux2 = None
#             score_aux3 = None
#             if self.cfg.WHICH_SCORE == 'main':
#                 score_aux1 = self.score_aux1(layer_3)
#                 score_aux2 = self.score_aux2(layer_2)
#                 score_aux3 = self.score_aux3(layer_1)
#             elif self.cfg.WHICH_SCORE == 'up':
#
#                 score_aux1 = self.score_aux1(torch.cat((up1_depth, up1_seg), 1))
#                 score_aux2 = self.score_aux2(torch.cat((up2_depth, up2_seg), 1))
#                 score_aux3 = self.score_aux3(torch.cat((up3_depth, up3_seg), 1))
#
#             score = F.interpolate(score_head, score_aux1.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux1
#             score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux2
#             score = F.interpolate(score, score_aux3.size()[2:], mode='bilinear', align_corners=True)
#             score = score + score_aux3
#
#             result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)
#
#             if cal_loss:
#                 result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         # if 'PIX2PIX' in self.cfg.LOSS_TYPES and phase == 'train':
#         #     result['loss_pix2pix_depth'] = self.pix2pix_criterion(result['gen_depth'], target_1)
#         #     result['loss_pix2pix_seg'] = self.pix2pix_criterion(result['gen_seg'], target_2)
#
#         return result


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


# #######################################################################
# class UNet(nn.Module):
#     def __init__(self, cfg, device=None):
#         super(UNet, self).__init__()
#
#         self.cfg = cfg
#         self.trans = not cfg.NO_TRANS
#         self.device = device
#         encoder = cfg.ARCH
#         num_classes = cfg.NUM_CLASSES
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if cfg.PRETRAINED == 'place':
#             resnet = resnet_models.__dict__['resnet18'](num_classes=365)
#             checkpoint = torch.load(resnet18_place_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place resnet18 loaded....')
#         else:
#             resnet = resnet18(pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#
#         self.score = nn.Conv2d(dims[1], num_classes, 1)
#
#         # norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
#         self.up1 = Conc_Up_Residual(dims[4], dims[3], norm=nn.BatchNorm2d)
#         self.up2 = Conc_Up_Residual(dims[3], dims[2], norm=nn.BatchNorm2d)
#         self.up3 = Conc_Up_Residual(dims[2], dims[1], norm=nn.BatchNorm2d)
#         self.up4 = Conc_Up_Residual(dims[1], dims[1], norm=nn.BatchNorm2d, conc_feat=False)
#
#         if pretrained:
#             init_weights(self.up1, 'normal')
#             init_weights(self.up2, 'normal')
#             init_weights(self.up3, 'normal')
#             init_weights(self.up4, 'normal')
#             init_weights(self.score, 'normal')
#
#         else:
#
#             init_weights(self, 'normal')
#
#     def set_cls_criterion(self, criterion):
#         self.cls_criterion = criterion.to(self.device)
#
#     def forward(self, source=None, label=None):
#         result = {}
#
#         layer_1 = self.layer1(self.relu(self.bn1(self.conv1(source))))
#         layer_2 = self.layer2(layer_1)
#         layer_3 = self.layer3(layer_2)
#         layer_4 = self.layer4(layer_3)
#
#         up1 = self.up1(layer_4, layer_3)
#         up2 = self.up2(up1, layer_2)
#         up3 = self.up3(up2, layer_1)
#         up4 = self.up4(up3)
#
#         result['cls'] = self.score(up4)
#         result['loss_cls'] = self.cls_criterion(result['cls'], label)
#
#         return result


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):

    def __init__(self, cfg, bins=(1, 2, 3, 6), dropout=0.1,
                 zoom_factor=8, use_ppm=True, pretrained=True, device=None):
        super(PSPNet, self).__init__()
        assert 2048 % len(bins) == 0
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.BatchNorm = batch_norm
        self.device = device
        self.trans = not cfg.NO_TRANS
        self.cfg = cfg
        dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        if self.trans:
            self.build_upsample_content_layers(dims)

        resnet = resnet_models.__dict__[cfg.ARCH](pretrained=pretrained, deep_base=True)
        print("load ", cfg.ARCH)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.trans:
            self.layer1._modules['0'].conv1.stride = (2, 2)
            if cfg.ARCH == 'resnet18':
                self.layer1._modules['0'].downsample = resnet.maxpool
            else:
                self.layer1._modules['0'].downsample._modules['0'].stride = (2, 2)
            self.build_upsample_content_layers(dims)

        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)
        # for n, m in self.layer4.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, self.BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            self.BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, cfg.NUM_CLASSES, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                self.BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, cfg.NUM_CLASSES, kernel_size=1)
            )

        init_type = 'normal'
        if self.trans:
            init_weights(self.up0, init_type)
            init_weights(self.up1, init_type)
            init_weights(self.up2, init_type)
            init_weights(self.up3, init_type)
            init_weights(self.up4, init_type)
            init_weights(self.up5, init_type)
            init_weights(self.up_seg, init_type)
            init_weights(self.score_head, init_type)
            init_weights(self.score_aux1, init_type)
            init_weights(self.score_aux2, init_type)

        init_weights(self.aux, init_type)
        init_weights(self.cls, init_type)
        init_weights(self.ppm, init_type)

    def build_upsample_content_layers(self, dims):

        norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        # norm = self.norm
        # self.up0 = Conc_Up_Residual_bottleneck(dims[7], dims[6], norm=norm, upsample=False)

        self.cross_1 = nn.Conv2d(dims[6], dims[4], kernel_size=1, bias=False)
        self.cross_2 = nn.Conv2d(dims[5], dims[4], kernel_size=1, bias=False)

        self.up1 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm, upsample=False)
        self.up2 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        self.up3 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        self.up4 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
        # self.up1 = Conc_Up_Residual_bottleneck(dims[6], dims[5], norm=norm)
        # self.up2 = Conc_Up_Residual_bottleneck(dims[5], dims[4], norm=norm)
        # self.up3 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        # self.up4 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        # self.up5 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm, conc_feat=False)

        self.up_seg = nn.Sequential(
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.score_aux1 = nn.Conv2d(1024, self.cfg.NUM_CLASSES, 1)
        self.score_aux2 = nn.Conv2d(512, self.cfg.NUM_CLASSES, 1)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_content_model(self, content_model):
        self.content_model = content_model.to(self.device)

    def forward(self, source, target=None, label=None, phase='train', content_layers=None, cal_loss=True, matrix=None,segSize=None):

        x = source
        y = label
        result = {}
        # x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        layer_0 = self.layer0(x)
        if not self.trans:
            layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        x = layer_4
        if self.use_ppm:
            x = self.ppm(x)

        if not self.trans:
            x = self.cls(x)
            if self.zoom_factor != 1:
                result['cls'] = F.interpolate(x, size=source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                aux = self.aux(layer_3)
                if self.zoom_factor != 1:
                    aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear', align_corners=True)
                main_loss = self.cls_criterion(result['cls'], y)
                aux_loss = self.cls_criterion(aux, y)
                result['loss_cls'] = main_loss + 0.4 * aux_loss

        else:
            # up0_seg = self.up0(x, layer_4)
            # up1_seg = self.up1(up0_seg, layer_3)
            # up2_seg = self.up2(up1_seg, layer_2)
            # up3_seg = self.up3(up2_seg, layer_1)
            # up4_seg = self.up4(up3_seg, layer_0)
            # up5_seg = self.up5(up4_seg)
            # up1_seg = self.up1(layer_4, layer_3)
            # up2_seg = self.up2(up1_seg, layer_2)
            # up3_seg = self.up3(up2_seg, layer_1)
            # up4_seg = self.up4(up3_seg, layer_0)
            # up5_seg = self.up5(up4_seg)

            cross_1 = self.cross_1(layer_4)
            cross_2 = self.cross_2(layer_3)
            cross_conc = torch.cat((cross_1, cross_2), 1)
            up1_seg = self.up1(cross_conc, layer_2)
            up2_seg = self.up2(up1_seg, layer_1)
            up3_seg = self.up3(up2_seg, layer_0)
            up4_seg = self.up4(up3_seg)

            result['gen_img'] = self.up_seg(up4_seg)

            score_aux1 = self.score_aux1(cross_conc)
            score_aux2 = self.score_aux2(up1_seg)

            x = self.cls(x)
            score = F.interpolate(x, score_aux1.size()[2:], mode='bilinear', align_corners=True)
            score = score + score_aux1 + score_aux2
            # score = F.interpolate(score, score_aux2.size()[2:], mode='bilinear', align_corners=True)
            # score = score + score_aux2
            result['cls'] = F.interpolate(score, source.size()[2:], mode='bilinear', align_corners=True)

            if cal_loss:
                # aux = self.aux(layer_3)
                # if self.zoom_factor != 1:
                #     aux = F.interpolate(aux, size=source.size()[2:], mode='bilinear', align_corners=True)
                # main_loss = self.cls_criterion(result['cls'], y)
                # aux_loss = self.cls_criterion(aux, y)
                # result['loss_cls'] = main_loss + 0.4 * aux_loss
                result['loss_cls'] = self.cls_criterion(result['cls'], y)
                result['loss_content'] = self.content_model(result['gen_img'], target, layers=content_layers)
        if segSize!=None:#new add
            result['cls'] = F.interpolate(result['cls'], segSize, mode='bilinear', align_corners=True)
            result['cls'] = nn.functional.softmax(result['cls'], dim=1)
        return result


##################### Recognition ############################
def upshuffle(in_planes, out_planes, upscale_factor, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        norm(out_planes),
        nn.ReLU(True)
    )


class TRecgNet_Scene_CLS(BaseTrans2Net_NoPooling):

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device=device)

        self.avg_pool_size = 14
        self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        norm = SynchronizedBatchNorm2d
        relu = nn.ReLU(True)

        # self.fc = Evaluator(cfg, fc_input_nc)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.in_features, cfg.NUM_CLASSES)
        )

        # self.fc_aux = Evaluator(cfg, fc_input_nc)
        # self.fc_aux = nn.Sequential(
        #     nn.AvgPool2d(7, 1),
        #     Flatten(),
        #     nn.Linear(512, 512),
        #     relu,
        #     nn.Linear(512, cfg.NUM_CLASSES)
        # )

        self.fc_aux = nn.Sequential(
            nn.AvgPool2d(7, 1),
            Flatten(),
            nn.Linear(512, cfg.NUM_CLASSES)
        )

        resnet = resnet_models.__dict__[self.cfg.ARCH](pretrained=True, deep_base=False)
        print('{0} pretrained:{1}'.format(self.cfg.ARCH, str(False)))

        self.maxpool = resnet.maxpool  # 1/4

        layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.cross_net = nn.Sequential(
            layer0, self.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # self.decoder_gen = Decoder_Gen_18(norm)
        # self.aspp = ASPP(512, 512, SynchronizedBatchNorm2d)
        # bins = (1, 2, 3, 6)
        # self.ppm = PPM(256, int(256 / len(bins)), bins, nn.BatchNorm2d)

        # self.cross_net = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        #     norm(64),
        #     relu,
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     norm(128),
        #     relu,
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     norm(256),
        #     relu,
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     norm(256),
        #     relu
        # )

        content_layers = cfg.CONTENT_LAYERS.split(',')
        if cfg.WHICH_CONTENT_NET == 'resnet18':
            dims_all = [64, 64, 128, 256, 512]
        else:
            dims_all = [64, 256, 512, 1024, 2048]
        dims_content = [dims_all[int(i)] for i in content_layers]
        # self.contrastive_net = Contrastive_Net(dims=dims_content)

        # self.contrastive_net = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=1),
        #     norm(1024),
        #     relu,
        #     nn.Conv2d(1024, 1024, kernel_size=1),
        #     norm(1024),
        #     relu,
        #     nn.Conv2d(1024, 1, kernel_size=1)
        # )
        # self.contrastive_net = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=1),
        #     norm(512),
        #     relu,
        #     nn.Conv2d(512, 512, kernel_size=1),
        #     norm(512),
        #     relu,
        #     nn.Conv2d(512, 1, kernel_size=1)
        # )
        # if 'alexnet' in cfg.ARCH:
        #     self.fc = nn.Sequential(
        #         self.maxpool,
        #         Flatten(),
        #         nn.Dropout(),
        #         nn.Linear(512 * 7 * 7, 4096),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(),
        #         nn.Linear(4096, 4096),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(4096, cfg.NUM_CLASSES)
        #     )
        # elif 'resnet' in cfg.ARCH:
        #     self.fc = nn.Sequential(
        #         Flatten(),
        #         nn.Linear(fc_input_nc, cfg.NUM_CLASSES)
        #     )
        # else:
        #     raise ValueError('cfg.ARCH is not recognized!!!')

        # self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        # self._up1 = upshuffle(512, 64, 8, norm=norm)
        # self._up2 = upshuffle(256, 64, 4, norm=norm)
        # self._up3 = upshuffle(128, 64, 2, norm=norm)
        # self._up4 = upshuffle(128, 64, 2)
        # self._up1 = upshuffle(256 + 64, 64, 8)
        # self._up2 = upshuffle(128 + 64, 64, 4)
        # self._up3 = upshuffle(64 + 64, 64, 2)
        # self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.smooth1 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1)
        # self.smooth4 = conv_norm_relu(64, 64, kernel_size=3, stride=1, padding=1)
        # self.smooth1 = nn.Conv2d(256 + 64, 64, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1)

        init_type = 'normal'
        if self.pretrained:

            for n, m in self.named_modules():
                if 'up' in n or 'fc' in n or 'smooth' in n or 'lat' in n or 'top' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

        set_criterion(cfg, self)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _upsample_conc(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(torch.cat([x, y], 1), size=(H, W), mode='bilinear', align_corners=True)

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):

        result = {}
        _, _, H, W = source.size()

        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        aspp = self.aspp(layer_4)
        result['feat'] = aspp

        result['avgpool'] = self.avgpool(aspp)
        # result['avgpool'] = self.avgpool(layer_4)

        if self.trans:

            # up1 = self.up1(layer_4)
            # up2 = self.up2(layer_3)
            # up3 = self.up3(layer_2)
            # vol = torch.cat([F.interpolate(d, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
            #                  for d in [up1, up2, up3]], dim=1)
            # up = self.up_image(vol)
            # result['gen_img'] = F.interpolate(up, size=(H, W), mode='bilinear', align_corners=True)

            gen, gen_decode = self.decoder_gen(layer_4, layer_1, layer_2, layer_3)
            result['gen_img'] = F.interpolate(gen, size=target.size()[2:], mode='bilinear', align_corners=True)

            # up = self.toplayer(layer_4)
            # lat1 = self.latlayer1(layer_3)
            # lat2 = self.latlayer2(layer_2)
            # lat3 = self.latlayer3(layer_1)
            #
            # up = self.up1(up, lat1)
            # up1 = up
            # up = self.up2(up, lat2)
            # up2 = up
            # up = self.up3(up, lat3)
            # up3 = up

            # up = self.up1(layer_4, layer_3)
            # up1 = up
            # up = self.up2(up, layer_2)
            # up2 = up
            # up = self.up3(up, layer_1)
            # up3 = up
            # vol = torch.cat([F.interpolate(d, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
            #                  for d in [up1, up2, up3]], dim=1)
            # up = self.up_image(vol)

            # img1 = F.interpolate(self.up_image_aux1(up1), size=(H, W), mode='bilinear', align_corners=True)
            # img2 = F.interpolate(self.up_image_aux2(up2), size=(H, W), mode='bilinear', align_corners=True)
            # img3 = F.interpolate(self.up_image_aux3(up3), size=(H, W), mode='bilinear', align_corners=True)
            # result['gen_img'] = F.interpolate(img1 + img2 + img3, size=(H, W), mode='bilinear', align_corners=True)
            # result['gen_img'] = torch.sum([F.interpolate(d, size=(H, W), mode='bilinear', align_corners=True)
            #                  for d in [img1, img2, img3]])

            # up = self.up1(layer_4, layer_3)
            # up1 = up
            # if self.cfg.MULTI_SCALE:
            #     up = self.up_image(up)
            #     result['gen_img'] = F.interpolate(up, target.size()[2:], mode='bilinear', align_corners=True)
            #
            # else:
            #     up = self.up2(up, layer_2)
            #     up2 = up
            #     up = self.up3(up, layer_1)
            #     up3 = up
            #     vol = torch.cat([F.upsample(d, size=(int(H/2), int(W/2)), mode='bilinear') for d in [up1, up2, up3]], dim=1)
            # # vol = torch.cat([F.upsample(d, size=(H, W), mode='bilinear') for d in [up0, up1, up2, up3]], dim=1)
            # # up = self.up_image(vol)
            # up = self.up_image(vol)
            # result['gen_img'] = F.interpolate(up, size=(H, W), mode='bilinear', align_corners=True)

            if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                out_content = self.content_model(result['gen_img'], target, layers=content_layers)
                if cal_loss:
                    result['loss_content'] = out_content['content_loss']

            if 'CONTRAST' in self.cfg.LOSS_TYPES and cal_loss:
                # if 'SEMANTIC' not in self.cfg.LOSS_TYPES:
                #     out_content = self.content_model(result['gen_img'], target, layers=content_layers)
                #
                feat_gen = self.cross_net(result['gen_img'])
                feat_target = self.cross_net(target)
                feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)
                # feat_gen = out_content['feat_gen']
                # feat_target = out_content['feat_target']
                # feat_target_neg = {k: torch.cat((f[1:], f[0].unsqueeze(0)), dim=0) for k, f in feat_target.items()}
                #
                # pos = [torch.cat([gen, feat_target[i]], 1) for i, gen in enumerate(feat_gen)]
                # neg = [torch.cat([gen, feat_target_neg[i]], 1) for i, gen in enumerate(feat_gen)]

                pos = torch.cat((feat_gen, feat_target), 1)
                neg = torch.cat((feat_gen, feat_target_neg), dim=1)

                # pos = [torch.cat([gen, feat_target[k]], 1) for k, gen in feat_gen.items()]
                # neg = [torch.cat([gen, feat_target_neg[k]], 1) for k, gen in feat_gen.items()]
                # pos = [torch.cat([gen, feat_target[i]], 1) for i, gen in enumerate(feat_gen)]
                # neg = [torch.cat([gen, feat_target_neg[i]], 1) for i, gen in enumerate(feat_gen)]

                # loss_contrast = []
                # for pos_f, neg_f in zip(pos, neg):
                #     Ej = F.softplus(-pos_f).mean()
                #     Em = F.softplus(neg_f).mean()
                #     # loss_contrast.append(Em - Ej)
                #     # Ej = F.softplus(-self.contrastive_net(pos_f)).mean()
                #     # Em = F.softplus(self.contrastive_net(neg_f)).mean()
                #     loss_contrast.append(Em + Ej)

                # Ej = F.softplus(-pos[-2])
                # Em = F.softplus(neg[-2])
                # Ej = F.softplus(-pos[-2])
                # Em = F.softplus(neg[-2])
                Ej = F.softplus(-self.contrastive_net(pos)).mean()
                Em = F.softplus(self.contrastive_net(neg)).mean()
                result['loss_contrast'] = Em + Ej

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

            if 'CLS' in self.cfg.LOSS_TYPES:
                #
                result['cls'] = self.fc(result['avgpool'])
                if cal_loss:
                    result['loss_cls'] = self.cls_criterion(result['cls'], label)

                    # if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                    #     alpha = 1
                    #     feat = out_content['feat_gen']['4']
                    #     fc_aux = self.fc_aux(feat)
                    #     # alpha = 0
                    #     # result['cls'] = result['cls'] * alpha + fc_aux * (1 - alpha)
                    #     result['loss_cls'] += self.cls_criterion(fc_aux, label) * alpha

                    # if 'PIX2PIX' in self.cfg.LOSS_TYPES:
                    #     feat = self.cross_net(result['gen_img'])
                    #     fc_aux = self.fc_aux(nn.AvgPool2d(7, 1)(feat))
                    #     result['loss_cls'] += self.cls_criterion(fc_aux, label)

        return result

# class LCM(nn.Module):
#     def __init__(self, in_dim, reduction_dim, in_dims=[128, 128, 512, 1024]):
#         super(LCM, self).__init__()
#         self.features = []
#         for bin in bins:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
#             ))
#         self.features = nn.ModuleList(self.features)
#
#     def forward(self, x):
#         x_size = x.size()
#         out = [x]
#         for f in self.features:
#             out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
#         return torch.cat(out, 1)

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck


class ResBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, norm=nn.BatchNorm2d):
        super(ResBottleneck, self).__init__()
        dim = out_channels // 2
        self.conv_reduce = conv_norm_relu(in_channels, dim, kernel_size=1, stride=1, padding=0, norm=norm)
        self.conv_conv = conv_norm_relu(dim, dim, kernel_size=3, stride=stride, norm=norm)
        self.conv_expand = conv_norm_relu(dim, out_channels, kernel_size=1, stride=1, padding=0, norm=norm)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck

class TrecgNet_Scene_CLS_Maxpool(BaseTrans2Net):

    def __init__(self, cfg, device=None):
        super(TrecgNet_Scene_CLS_Maxpool, self).__init__(cfg, device)
        self.avgpool = nn.AvgPool2d(7, 1)
        if self.cfg.ARCH == 'resnet18' or 'alexnet' in self.cfg.ARCH:
            self.fc_input_nc = 512
        else:
            self.fc_input_nc = 2048

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.fc_input_nc, cfg.NUM_CLASSES)
        )

        self.fc_aux_semantic = nn.Sequential(
            Flatten(),
            nn.Linear(self.fc_input_nc, cfg.NUM_CLASSES)
        )

        self.fc_aux = nn.Sequential(
            Flatten(),
            nn.Linear(int(self.fc_input_nc), cfg.NUM_CLASSES)
        )

        content_layers = cfg.CONTENT_LAYERS.split(',')
        if cfg.WHICH_CONTENT_NET == 'resnet18':
            dims_all = [64, 64, 128, 256, 512]
        else:
            dims_all = [64, 256, 512, 1024, 2048]
        dims_content = [dims_all[int(i)] for i in content_layers]
        self.contrastive_net = Contrastive_Net(dims=dims_content)

        self.aspp = ASPP(dim_in=dims_all[-1], dim_out=dims_all[-1] // 4, init_type='normal')

        self.norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        if self.trans:
            self.build_upsample_content_layers(dims_all)

        init_type = 'normal'
        if self.pretrained:

            for n, m in self.named_modules():
                if 'up' in n or 'fc' in n or 'skip' in n or 'aux' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

        set_criterion(cfg, self)

    def build_upsample_content_layers(self, dims):

        lat_dim = dims[-1] // 4
        self.up32x = ConvNormReLULayer(dims[-1], dims[-2], kernel_size=1, padding=0, norm=self.norm)
        self.up16x = ConvNormReLULayer(dims[-2] + lat_dim, dims[-3], kernel_size=1, padding=0, norm=self.norm)
        self.up8x = ConvNormReLULayer(dims[-3] + lat_dim, dims[-4], kernel_size=1, padding=0, norm=self.norm)
        self.up4x = ConvNormReLULayer(dims[-4] + lat_dim, dims[-5], kernel_size=1, padding=0, norm=self.norm)

        self.lat16x = ConvNormReLULayer(dims[-2], lat_dim, kernel_size=3, padding=1, norm=self.norm)
        self.lat8x = ConvNormReLULayer(dims[-3], lat_dim, kernel_size=3, padding=1, norm=self.norm)
        self.lat4x = ConvNormReLULayer(dims[-4], lat_dim, kernel_size=3, padding=1, norm=self.norm)

        self.smooth16x = ConvNormReLULayer(dims[-3], dims[-3], kernel_size=3, padding=1, norm=self.norm)
        self.smooth8x = ConvNormReLULayer(dims[-4], dims[-4], kernel_size=3, padding=1, norm=self.norm)
        self.smooth4x = ConvNormReLULayer(dims[-5], dims[-5], kernel_size=3, padding=1, norm=self.norm)

        self.up_image = nn.Sequential(
            nn.Conv2d(dims[-5], 3, 1),
            # nn.Conv2d(dims[-5], 3, 7, 1, 3, bias=False),
            # nn.Tanh()
        )

    def set_sample_model(self, sample_model):
        self.sample_model = sample_model
        self.compl_net = sample_model.compl_net
        # fix_grad(self.compl_net)
        # import util.utils as util
        # cls_criterion = util.CrossEntropyLoss(weight=self.cfg.CLASS_WEIGHTS_TRAIN, device=self.device,
        #                                           ignore_index=self.cfg.IGNORE_LABEL)
        # self.compl_net.cls_criterion = cls_criterion.to(self.device)
        self.compl_net.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.fc_input_nc, self.cfg.NUM_CLASSES)
        )
        # self.fc_conc = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(self.fc_input_nc * 2, self.fc_input_nc),
        #     nn.ReLU(True),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.fc_input_nc, self.cfg.NUM_CLASSES)
        # )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        _, _, H, W = source.size()
        x_size = source.size()
        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        result['feat'] = layer_4
        result['avgpool'] = self.avgpool(layer_4)

        # m1f = F.interpolate(layer_0, source.size()[2:], mode='bilinear', align_corners=True)
        # im_arr = source.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        # canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        # for i in range(x_size[0]):
        #     canny[i] = cv2.Canny(im_arr[i], 10, 100)
        # canny = torch.from_numpy(canny).cuda().float()

        lat1 = self.lat16x(layer_3)
        lat2 = self.lat8x(layer_2)
        lat3 = self.lat4x(layer_1)
        # lat4 = self.latlayer4(layer_0)

        x = layer_4
        if self.trans:

            x = self.up32x(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, lat1], dim=1)
            x = self.up16x(x)
            x = self.smooth16x(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, lat2], dim=1)
            x = self.up8x(x)
            x = self.smooth8x(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, lat3], dim=1)
            x = self.up4x(x)
            x = self.smooth4x(x)
            x = self.up_image(x)
            result['gen_img'] = F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True)
            out_content = None

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                out_content = self.content_model(result['gen_img'], target, layers=content_layers)
                result['loss_content'] = out_content['content_loss']

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

            if 'CONTRAST' in self.cfg.LOSS_TYPES and cal_loss:

                # if out_content is None:
                #     out_content = self.content_model(result['gen_img'], target, layers=content_layers)
                #
                # feat_gen = out_content['feat_gen']
                # feat_target = out_content['feat_target']
                feat_target_neg = {k: torch.cat((f[1:], f[0].unsqueeze(0)), dim=0) for k, f in feat_target.items()}

                pos = [torch.cat([gen, feat_target[k]], 1) for k, gen in feat_gen.items()]
                neg = [torch.cat([gen, feat_target_neg[k]], 1) for k, gen in feat_gen.items()]
                result['feat_gen'] = feat_gent
                result['feat_target'] = feat_target
                result['loss_contrast'] = self.contrastive_net(pos, neg)

        if 'CLS' in self.cfg.LOSS_TYPES:

            # result['cls'] = self.fc(result['feat'])
            result['cls'] = self.fc(result['avgpool'])

            if self.cfg.USE_COMPL_DATA:
                with torch.no_grad():
                    result_sample = self.sample_model(source, target, label, phase, content_layers, cal_loss=False)
                input_compl = result_sample['gen_img'].detach()
                result_compl = self.compl_net(input_compl, label=label, cal_loss=True)
                result['compl_source'] = input_compl

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)
                # if self.trans:
                #     feat = torch.cat([out_content['feat_gen']['3'], up1], 1)
                #     fc_aux = self.fc_aux(nn.AvgPool2d(14, 1)(feat))
                #     result['loss_cls'] += self.cls_criterion(fc_aux, label)

                if self.cfg.USE_COMPL_DATA:
                    result['loss_cls_compl'] = result_compl['loss_cls']

                # if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                #     feat = out_content['feat_gen']['4']
                #     fc_aux = self.fc_aux_semantic(nn.AvgPool2d(7, 1)(feat))
                #     # alpha = 0
                #     # result['cls'] = result['cls'] * alpha + fc_aux * (1 - alpha)
                #     result['loss_cls'] += self.cls_criterion(fc_aux, label)

            if self.cfg.USE_COMPL_DATA:
                alpha_main = 0.6
                result['cls'] = result['cls'] * alpha_main + result_compl['cls'] * (1 - alpha_main)

        return result


class Contrastive_RGBD(nn.Module):

    def __init__(self, cfg, device=None):
        super(Contrastive_RGBD, self).__init__()
        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        self.arch = cfg.ARCH

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            self.pretrained = True
        else:
            self.pretrained = False

        self.rgb = resnet_models.__dict__[self.arch](pretrained=False)
        self.depth = resnet_models.__dict__[self.arch](pretrained=False)

        self.avgpool = nn.AvgPool2d(7, 1)
        if self.cfg.ARCH == 'resnet18' or 'alexnet' in self.cfg.ARCH:
            self.fc_input_nc = 512
        else:
            self.fc_input_nc = 2048

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.fc_input_nc, cfg.NUM_CLASSES)
        )

        if cfg.WHICH_CONTENT_NET == 'resnet18':
            dims_all = [64, 64, 128, 256, 512]
        else:
            dims_all = [64, 256, 512, 1024, 2048]

        self.contrastive_net = Layer_Wise_Contrastive_Net(dims=dims_content)

        self.norm = nn.InstanceNorm2d if self.cfg.UPSAMPLE_NORM == 'in' else nn.BatchNorm2d
        if self.trans:
            self.build_upsample_content_layers(dims_all)

        init_type = 'normal'
        if self.pretrained:

            for n, m in self.named_modules():
                if 'up' in n or 'fc' in n or 'skip' in n or 'aux' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

        set_criterion(cfg, self)

    def build_upsample_content_layers(self, dims):

        lat_dim = dims[-1] // 4
        self.up32x = ConvNormReLULayer(dims[-1], dims[-2], kernel_size=1, padding=0, norm=self.norm)
        self.up16x = ConvNormReLULayer(dims[-2] + lat_dim, dims[-3], kernel_size=1, padding=0, norm=self.norm)
        self.up8x = ConvNormReLULayer(dims[-3] + lat_dim, dims[-4], kernel_size=1, padding=0, norm=self.norm)
        self.up4x = ConvNormReLULayer(dims[-4] + lat_dim, dims[-5], kernel_size=1, padding=0, norm=self.norm)

        self.lat16x = ConvNormReLULayer(dims[-2], lat_dim, kernel_size=3, padding=1, norm=self.norm)
        self.lat8x = ConvNormReLULayer(dims[-3], lat_dim, kernel_size=3, padding=1, norm=self.norm)
        self.lat4x = ConvNormReLULayer(dims[-4], lat_dim, kernel_size=3, padding=1, norm=self.norm)

        self.smooth16x = ConvNormReLULayer(dims[-3], dims[-3], kernel_size=3, padding=1, norm=self.norm)
        self.smooth8x = ConvNormReLULayer(dims[-4], dims[-4], kernel_size=3, padding=1, norm=self.norm)
        self.smooth4x = ConvNormReLULayer(dims[-5], dims[-5], kernel_size=3, padding=1, norm=self.norm)

        self.up_image = nn.Sequential(
            nn.Conv2d(dims[-5], 3, 1),
            # nn.Conv2d(dims[-5], 3, 7, 1, 3, bias=False),
            # nn.Tanh()
        )

    def set_sample_model(self, sample_model):
        self.sample_model = sample_model
        self.compl_net = sample_model.compl_net
        # fix_grad(self.compl_net)
        # import util.utils as util
        # cls_criterion = util.CrossEntropyLoss(weight=self.cfg.CLASS_WEIGHTS_TRAIN, device=self.device,
        #                                           ignore_index=self.cfg.IGNORE_LABEL)
        # self.compl_net.cls_criterion = cls_criterion.to(self.device)
        self.compl_net.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.fc_input_nc, self.cfg.NUM_CLASSES)
        )
        # self.fc_conc = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(self.fc_input_nc * 2, self.fc_input_nc),
        #     nn.ReLU(True),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.fc_input_nc, self.cfg.NUM_CLASSES)
        # )

    def forward(self, source=None, target=None, label=None, phase='train', content_layers=None, cal_loss=True):
        result = {}

        _, _, H, W = source.size()
        x_size = source.size()
        layer_0 = self.layer0(source)
        layer_1 = self.layer1(layer_0)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)
        result['feat'] = layer_4
        result['avgpool'] = self.avgpool(layer_4)

        # m1f = F.interpolate(layer_0, source.size()[2:], mode='bilinear', align_corners=True)
        # im_arr = source.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        # canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        # for i in range(x_size[0]):
        #     canny[i] = cv2.Canny(im_arr[i], 10, 100)
        # canny = torch.from_numpy(canny).cuda().float()

        lat1 = self.lat16x(layer_3)
        lat2 = self.lat8x(layer_2)
        lat3 = self.lat4x(layer_1)
        # lat4 = self.latlayer4(layer_0)

        x = layer_4
        if self.trans:

            x = self.up32x(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, lat1], dim=1)
            x = self.up16x(x)
            x = self.smooth16x(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, lat2], dim=1)
            x = self.up8x(x)
            x = self.smooth8x(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, lat3], dim=1)
            x = self.up4x(x)
            x = self.smooth4x(x)
            x = self.up_image(x)
            result['gen_img'] = F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True)
            out_content = None

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
                out_content = self.content_model(result['gen_img'], target, layers=content_layers)
                result['loss_content'] = out_content['content_loss']

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and cal_loss:
                result['loss_pix2pix'] = self.pix2pix_criterion(result['gen_img'], target)

            if 'CONTRAST' in self.cfg.LOSS_TYPES and cal_loss:

                # if out_content is None:
                #     out_content = self.content_model(result['gen_img'], target, layers=content_layers)
                #
                # feat_gen = out_content['feat_gen']
                # feat_target = out_content['feat_target']
                feat_target_neg = {k: torch.cat((f[1:], f[0].unsqueeze(0)), dim=0) for k, f in feat_target.items()}

                pos = [torch.cat([gen, feat_target[k]], 1) for k, gen in feat_gen.items()]
                neg = [torch.cat([gen, feat_target_neg[k]], 1) for k, gen in feat_gen.items()]
                result['feat_gen'] = feat_gen
                result['feat_target'] = feat_target
                result['loss_contrast'] = self.contrastive_net(pos, neg)

        if 'CLS' in self.cfg.LOSS_TYPES:

            # result['cls'] = self.fc(result['feat'])
            result['cls'] = self.fc(result['avgpool'])

            if self.cfg.USE_COMPL_DATA:
                with torch.no_grad():
                    result_sample = self.sample_model(source, target, label, phase, content_layers, cal_loss=False)
                input_compl = result_sample['gen_img'].detach()
                result_compl = self.compl_net(input_compl, label=label, cal_loss=True)
                result['compl_source'] = input_compl

            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)
                # if self.trans:
                #     feat = torch.cat([out_content['feat_gen']['3'], up1], 1)
                #     fc_aux = self.fc_aux(nn.AvgPool2d(14, 1)(feat))
                #     result['loss_cls'] += self.cls_criterion(fc_aux, label)

                if self.cfg.USE_COMPL_DATA:
                    result['loss_cls_compl'] = result_compl['loss_cls']

                # if 'SEMANTIC' in self.cfg.LOSS_TYPES:
                #     feat = out_content['feat_gen']['4']
                #     fc_aux = self.fc_aux_semantic(nn.AvgPool2d(7, 1)(feat))
                #     # alpha = 0
                #     # result['cls'] = result['cls'] * alpha + fc_aux * (1 - alpha)
                #     result['loss_cls'] += self.cls_criterion(fc_aux, label)

            if self.cfg.USE_COMPL_DATA:
                alpha_main = 0.6
                result['cls'] = result['cls'] * alpha_main + result_compl['cls'] * (1 - alpha_main)

        return result


class Fusion(nn.Module):

    def __init__(self, cfg, rgb_model=None, depth_model=None, device='cuda'):
        super(Fusion, self).__init__()
        self.cfg = cfg
        self.device = device
        # self.rgb_model = rgb_model
        # self.depth_model = depth_model
        self.net_RGB = rgb_model
        self.net_depth = depth_model
        # self.net_RGB = self.construct_single_modal_net(rgb_model.source_net)
        # self.net_depth = self.construct_single_modal_net(depth_model.source_net)

        if cfg.FIX_GRAD:
            fix_grad(self.net_RGB)
            fix_grad(self.net_depth)

        self.avgpool = nn.AvgPool2d(14, 1)
        # self.fc = nn.Linear(1024 * 4, cfg.NUM_CLASSES)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, cfg.NUM_CLASSES)
        )

        set_criterion(cfg, self)

        init_weights(self.fc, 'normal')

    # only keep the classification branch
    def construct_single_modal_net(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module

        ops = [model.conv1, model.bn1, model.relu, model.layer1, model.layer2,
               model.layer3, model.layer4]
        return nn.Sequential(*ops)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, input_rgb, input_depth, label, phase=None, cal_loss=True):

        result = {}
        rgb_specific = self.net_RGB(input_rgb, cal_loss=False)
        depth_specific = self.net_depth(input_depth, cal_loss=False)
        # self.smooth = conv_norm_relu(1024, 512)

        # rgb = self.avgpool(rgb_specific)
        # rgb = rgb.view(rgb.size(0), -1)
        # cls_rgb = self.rgb_model.fc(rgb)
        # out['cls'] = self.rgb_model.fc(x)

        # depth = self.avgpool(depth_specific)
        # depth = depth.view(depth.size(0), -1)
        # cls_depth = self.depth_model.fc(depth)
        # out['cls'] = self.depth_model.fc(x)

        # cls_rgb = rgb_specific['cls']
        # cls_depth = depth_specific['cls']

        x = torch.cat((rgb_specific['feat'], depth_specific['feat']), 1).to(self.device)
        x = self.avgpool(x)
        result['cls'] = self.fc(x)

        # alpha = 0.65
        # result['cls'] = alpha * cls_rgb + (1-alpha) * cls_depth

        if cal_loss:

            # if 'SEMANTIC' in self.cfg.LOSS_TYPES and target is not None and phase == 'train':
            #     loss_content = self.content_model(out['gen_img'], target, layers=content_layers) * self.cfg.ALPHA_CONTENT

            if 'CLS' in self.cfg.LOSS_TYPES and not self.cfg.UNLABELED:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


class Fusion_Trecg_Fake(nn.Module):

    def __init__(self, cfg, rgb_model=None, depth_model=None, device='cuda'):
        super(Fusion_Trecg_Fake, self).__init__()
        self.cfg = cfg
        self.device = device
        self.rgb_model = rgb_model
        self.depth_model = depth_model

        fix_grad(self.rgb_model)
        fix_grad(self.depth_model)
        # fix_grad(self.rgb_model.compl_net)
        # fix_grad(self.depth_model.compl_net)

        # for n, m in self.rgb_model.named_modules():
        #     if 'up' in n:
        #         fix_grad(m)
        # for n, m in self.depth_model.named_modules():
        #     if 'up' in n:
        #         fix_grad(m)

        # self.net_RGB = self.construct_single_modal_net(rgb_model)
        # self.net_depth = self.construct_single_modal_net(depth_model)
        # self.rgb_net = self.rgb_model.net
        # self.rgb_compl_net = self.rgb_model.compl_net
        # self.depth_net = self.depth_model.net
        # self.depth_compl_net = self.depth_model.compl_net

        self.avgpool = nn.AvgPool2d(14, 1)

        # self.fc = nn.Linear(2048, cfg.NUM_CLASSES)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.cfg.NUM_CLASSES * 4, self.cfg.NUM_CLASSES),
        # )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, cfg.NUM_CLASSES)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, cfg.NUM_CLASSES)
        # )

        # self.conc_convs = nn.Sequential(
        #     conv_norm_relu(2048, 1024, kernel_size=1, padding=0),
        #     conv_norm_relu(1024, 512, kernel_size=3, padding=1)
        # )

        self.flatten = flatten

        init_weights(self.fc, 'normal')

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_content_model(self, content_model):
        self.content_model = content_model

    def forward(self, input_rgb, input_depth, label, phase=None, cal_loss=True):

        result = {}

        rgb_result = self.rgb_model(input_rgb, cal_loss=False)
        depth_result = self.depth_model(input_depth, cal_loss=False)

        result['gen_depth'] = rgb_result['gen_img']
        result['gen_rgb'] = depth_result['gen_img']
        result['gen_img'] = depth_result['gen_img']
        # rgb_cls = self.flatten(rgb_result['cls_original'])
        # rgb_cls_compl = self.flatten(rgb_result['cls_compl'])
        # depth_cls = self.flatten(depth_result['cls_original'])
        # depth_cls_compl = self.flatten(depth_result['cls_compl'])

        if 'SEMANTIC' in self.cfg.LOSS_TYPES and cal_loss:
            result['loss_content'] = self.content_model(rgb_result['gen_img'], input_depth)
            result['loss_content'] += self.content_model(depth_result['gen_img'], input_rgb)

        # cls = torch.cat([rgb_cls, rgb_cls_compl, depth_cls, depth_cls_compl], 1)
        rgb_feat = rgb_result['feat']
        rgb_feat_compl = rgb_result['feat']
        depth_feat = depth_result['feat']
        depth_feat_compl = depth_result['feat']
        feat_conc = torch.cat([rgb_feat, rgb_feat_compl, depth_feat, depth_feat_compl], 1)

        # feat = self.conc_convs(feat_conc)
        result['cls'] = self.fc(flatten(self.avgpool(feat_conc)))


        # rgb_cls = self.flatten(rgb_result['avgpool'])
        # rgb_cls_compl = self.flatten(rgb_result['avgpool_compl'])
        # depth_cls = self.flatten(depth_result['avgpool'])
        # depth_cls_compl = self.flatten(depth_result['avgpool_compl'])
        # cls = torch.cat([rgb_cls, rgb_cls_compl, depth_cls, depth_cls_compl], 1)
        # self.smooth = conv_norm_relu(1024, 512)
        # rgb = self.avgpool(rgb_specific)
        # rgb = rgb.view(rgb.size(0), -1)
        # cls_rgb = self.rgb_model.fc(rgb)11Kkk
        # out['cls'] = self.rgb_model.fc(x)

        # depth = self.avgpool(depth_specific)
        # depth = depth.view(depth.size(0), -1)
        # cls_depth = self.depth_model.fc(depth)
        # out['cls'] = self.depth_model.fc(x)

        # alpha = 0.6
        # out['cls'] = alpha * cls_rgb + (1 - alpha) * cls_depth

        # result['cls'] = (rgb_cls + rgb_cls_compl + depth_cls + depth_cls_compl) * 0.25
        # result['cls'] = rgb_cls * 0.4 + rgb_cls_compl * 0.1 + depth_cls * 0.5 + depth_cls_compl * 0.1
        # result['cls'] = self.fc(cls)
        if cal_loss:

            # if 'SEMANTIC' in self.cfg.LOSS_TYPES and target is not None and phase == 'train':
            #     loss_content = self.content_model(out['gen_img'], target, layers=content_layers) * self.cfg.ALPHA_CONTENT

            if 'CLS' in self.cfg.LOSS_TYPES and not self.cfg.UNLABELED:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)

        return result


########################### INFOMAX ###############################
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Source_Model(BaseTrans2Net):

    def __init__(self, cfg, device=None):
        super(Source_Model, self).__init__(cfg, device)

        #
        # self.maxpool = resnet.maxpool  # 1/4
        #
        # if self.encoder == 'resnet18':
        #     self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # else:
        #     self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
        #                                 resnet.conv3, resnet.bn3, resnet.relu)
        # # self.layer0 = nn.Sequential(
        # #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        # #     nn.BatchNorm2d(64),
        # #     nn.ReLU(inplace=True)
        # # )
        # # self.layer1 = nn.Sequential(
        # #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        # #     nn.BatchNorm2d(64),
        # #     nn.ReLU(inplace=True)
        # # )
        # # self.layer2 = nn.Sequential(
        # #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        # #     nn.BatchNorm2d(128),
        # #     nn.ReLU(inplace=True)
        # # )
        # # self.layer3 = nn.Sequential(
        # #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        # #     nn.BatchNorm2d(256),
        # #     nn.ReLU(inplace=True)
        # # )
        # # self.layer4 = nn.Sequential(
        # #     nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        # #     nn.BatchNorm2d(512),
        # #     nn.ReLU(inplace=True)
        # # )
        # self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 1/4
        # self.layer2 = resnet.layer2  # 1/8
        # self.layer3 = resnet.layer3  # 1/16
        # self.layer4 = resnet.layer4  # 1/32
        # self.avg_pool = nn.AvgPool2d(7, 1)

        init_type = 'normal'
        if self.pretrained:

            for n, m in self.named_modules():
                if 'up' in n or 'fc' in n or 'skip' in n or 'aux' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

        set_criterion(cfg, self)
        self.evaluator = nn.Sequential(
            Flatten(),
            nn.Linear(self.resnet.fc.in_features, cfg.NUM_CLASSES)
        )
        # self.evaluator = Evaluator(cfg, input_nc=self.resnet.fc.in_features)
        # self.fc_z = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(512 * 7 * 7, 128)
        # )

        # if self.trans or self.cfg.FT or self.cfg.RESUME:
        #     self.layer1._modules['0'].conv1.stride = (2, 2)
        #     if cfg.ARCH == 'resnet18':
        #         self.layer1._modules['0'].downsample = resnet.maxpool
        #     else:
        #         self.layer1._modules['0'].downsample._modules['0'].stride = (2, 2)

        # if self.trans:
        #     self.build_upsample_layers()

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def forward(self, x, target=None, label=None, class_only=False):
        out = {}

        layer_0 = self.layer0(x)
        # if not self.trans and not self.cfg.FT and not self.cfg.RESUME:
        #     layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        out['feat_1'] = layer_1
        layer_2 = self.layer2(layer_1)
        out['feat_2'] = layer_2
        layer_3 = self.layer3(layer_2)
        out['feat_3'] = layer_3
        layer_4 = self.layer4(layer_3)
        out['feat_4'] = layer_4
        out['avgpool'] = self.avgpool(layer_4)
        # out['z'] = self.fc_z(layer_4)

        if class_only:
            out['pred'] = self.evaluator(out['avgpool'])
            return out

        if label is not None:
            out['pred'] = self.evaluator(out['avgpool'])
            out['cls_loss'] = self.cls_criterion(out['pred'], label)

        # if class_only:
        #
        #     lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator(avg_rgb)
        #     out['pred'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
        #     return out
        #
        # if label is not None:
        #     lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator(avg_rgb)
        #     out['pred'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
        #     out['cls_loss'] = self.cls_criterion(lgt_glb_mlp_rgb, label) + self.cls_criterion(lgt_glb_lin_rgb, label)

        if self.trans:
            up = self.up1(layer_4, layer_3)
            up = self.up2(up, layer_2)
            if self.cfg.MULTI_SCALE:
                up = self.up_image(up)
            else:
                up = self.up3(up, layer_1)
                up = self.up4(up, layer_0)
                up = self.up_image(up)

            out['gen_cross'] = F.interpolate(up, target.size()[2:], mode='bilinear', align_corners=True)

        return out


class Source_Model_Fusion(BaseTrans2Net):

    def __init__(self, cfg, device=None):
        super(Source_Model_Fusion, self).__init__(cfg, device)

        init_type = 'normal'
        if self.pretrained:

            for n, m in self.named_modules():
                if 'up' in n or 'fc' in n or 'skip' in n or 'aux' in n:
                    init_weights(m, init_type)
        else:
            init_weights(self, init_type)

        set_criterion(cfg, self)
        self.evaluator = nn.Sequential(
            Flatten(),
            nn.Linear(self.resnet.fc.in_features, cfg.NUM_CLASSES)
        )

        self.z_rgb = nn.Sequential(
            Flatten(),
            nn.Linear(512 * 7 * 7, 128)
        )

        self.z_depth = nn.Sequential(
            Flatten(),
            nn.Linear(512 * 7 * 7, 128)
        )

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def forward(self, x, target=None, label=None, class_only=False):
        out = {}

        layer_0 = self.layer0(x)
        # if not self.trans and not self.cfg.FT and not self.cfg.RESUME:
        #     layer_0 = self.maxpool(layer_0)
        layer_1 = self.layer1(layer_0)
        out['feat_1'] = layer_1
        layer_2 = self.layer2(layer_1)
        out['feat_2'] = layer_2
        layer_3 = self.layer3(layer_2)
        out['feat_3'] = layer_3
        layer_4 = self.layer4(layer_3)
        out['feat_4'] = layer_4
        out['avgpool'] = self.avgpool(layer_4)
        # out['z'] = self.fc_z(layer_4)

        if class_only:
            out['pred'] = self.evaluator(out['avgpool'])
            return out

        if label is not None:
            out['pred'] = self.evaluator(out['avgpool'])
            out['cls_loss'] = self.cls_criterion(out['pred'], label)

        # if class_only:
        #
        #     lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator(avg_rgb)
        #     out['pred'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
        #     return out
        #
        # if label is not None:
        #     lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator(avg_rgb)
        #     out['pred'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
        #     out['cls_loss'] = self.cls_criterion(lgt_glb_mlp_rgb, label) + self.cls_criterion(lgt_glb_lin_rgb, label)

        if self.trans:
            up = self.up1(layer_4, layer_3)
            up = self.up2(up, layer_2)
            if self.cfg.MULTI_SCALE:
                up = self.up_image(up)
            else:
                up = self.up3(up, layer_1)
                up = self.up4(up, layer_0)
                up = self.up_image(up)

            out['gen_cross'] = F.interpolate(up, target.size()[2:], mode='bilinear', align_corners=True)

        return out


class Cross_Model(nn.Module):

    def __init__(self, cfg, device=None):
        super(Cross_Model, self).__init__()

        self.cfg = cfg
        self.trans = not cfg.NO_TRANS
        self.device = device
        relu = nn.ReLU(True)
        norm = nn.BatchNorm2d

        resnet = resnet_models.__dict__[self.cfg.ARCH](pretrained=True, deep_base=False)
        print('{0} pretrained:{1}'.format(self.cfg.ARCH, str(False)))

        self.maxpool = resnet.maxpool  # 1/4

        layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.model = nn.Sequential(
            layer0, self.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # self.model = models_tv.vgg11(pretrained=True).features

        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        #     norm(64),
        #     relu,
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     norm(128),
        #     relu,
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     norm(256),
        #     relu,
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     norm(256),
        #     relu
        # )

        # self.model = nn.Sequential(
        #     layer0,
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     norm(128),
        #     relu,
        #     models.BasicBlock(128, 128),
        #     models.BasicBlock(128, 128),
        #     models.BasicBlock(128, 128),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     norm(256),
        #     relu
        # )

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(64)
        # )
        #
        # self.conv2 = nn.Sequential(
        #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128)
        # )
        #
        # self.conv3 = nn.Sequential(
        #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(256)
        # )
        #
        # self.conv4 = nn.Sequential(
        #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(512)
        #     nn.ReLU(inplace=True)
        # )
        #
        # # self.conv5 = nn.Sequential(
        # #     nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        # #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # #     # nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        # #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # #     # nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        # #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # #     # nn.BatchNorm2d(512),
        # #     nn.ReLU(inplace=True))
        #
        # self.model = nn.Sequential(
        #     self.conv1, self.conv2, self.conv3, self.conv4
        # )

        self.fc_z = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 28 * 28, 128)
        )

        self.d_cross = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            relu,
            nn.Conv2d(1024, 1024, kernel_size=1),
            relu,
            nn.Conv2d(1024, 1, kernel_size=1)
        )

        self.fc_aux = nn.Sequential(
            Flatten(),
            nn.Linear(512, cfg.NUM_CLASSES)
        )

        # init_weights(self, 'normal')

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def forward(self, x, target, label=None):
        out = {}

        feat_gen = self.model(x)
        feat_target = self.model(target)
        feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)

        # z_gen = self.fc_z(feat_gen)
        # z_target = self.fc_z(feat_target)

        out['feat_gen'] = feat_gen
        out['feat_target'] = feat_target

        if 'CROSS' in self.cfg.LOSS_TYPES:

            pos = torch.cat((feat_gen, feat_target), 1)
            neg = torch.cat((feat_gen, feat_target_neg), dim=1)

            Ej = F.softplus(-self.d_cross(pos)).mean()
            Em = F.softplus(self.d_cross(neg)).mean()
            out['cross_loss'] = (Em + Ej)

        if 'HOMO' in self.cfg.LOSS_TYPES:

            out['homo_loss'] = nn.L1Loss()(feat_gen, feat_target)

            # layers = layers
            # if layers is None or not layers:
            #     layers = self.cfg.CONTENT_LAYERS.split(',')
            #
            # input_features = self.model((x + 1) / 2, layers)
            # target_targets = self.model((target + 1) / 2, layers)
            # len_layers = len(layers)
            # loss_fns = [self.criterion] * len_layers
            # alpha = [1] * len_layers
            #
            # content_losses = [alpha[i] * loss_fns[i](gen_content, target_targets[i])
            #                   for i, gen_content in enumerate(input_features)]
            # loss = sum(content_losses)

            # self_pos = torch.cat((feat_gen, feat_gen), 1)
            # feat_self_neg = torch.cat((feat_gen[1:], feat_gen[0].unsqueeze(0)), dim=0)
            # self_neg = torch.cat((feat_gen, feat_self_neg), dim=1)

            # Ej_self = -F.softplus(-self.d_cross(self_pos)).mean()
            # Em_self = F.softplus(self.d_cross(self_neg)).mean()
            # out['cross_loss_self'] = (Em_self - Ej_self)

        if 'PIX2PIX' in self.cfg.LOSS_TYPES:
            out['pix2pix_loss'] = self.pix2pix_criterion(x, target)

        if 'CLS' in self.cfg.LOSS_TYPES:
            out['pred'] = self.fc_aux(nn.AvgPool2d(7, 1)(out['feat_gen']))
            out['cls_loss'] = self.cls_criterion(out['pred'], label)

        return out


class Comparison_Net(nn.Module):

    def __init__(self):
        super(Comparison_Net, self).__init__()

        relu = nn.ReLU(True)
        norm = nn.BatchNorm2d
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # norm(64),
            relu,
        )

        self.d_cross = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            relu,
            nn.Conv2d(128, 1, kernel_size=1)
        )

        init_weights(self, 'normal')

    def forward(self, x, target):
        out = {}

        feat_gen = self.feat_extract(x)
        feat_target = self.feat_extract(target)
        feat_target_neg = torch.cat((feat_target[1:], feat_target[0].unsqueeze(0)), dim=0)

        out['feat_gen'] = feat_gen
        out['feat_target'] = feat_target

        pos = torch.cat((feat_gen, feat_target), 1)
        neg = torch.cat((feat_gen, feat_target_neg), dim=1)

        Ej = F.softplus(-self.d_cross(pos))
        Em = F.softplus(self.d_cross(neg))
        out['loss_contrast'] = (Em + Ej).mean()

        return out

def flatten(x):
    return x.reshape(x.size(0), -1)


class Evaluator(nn.Module):
    def __init__(self, cfg, input_nc=512):
        super(Evaluator, self).__init__()
        self.block_glb_mlp = MLPClassifier(input_nc, cfg.NUM_CLASSES, n_hidden=input_nc * 2, p=0.2)
        self.is_ft = cfg.FT
        # self.block_glb_lin = \
        #     MLPClassifier(512, self.n_classes, n_hidden=None, p=0.0)

    def forward(self, ftr_1):
        '''
        Input:
          ftr_1 : features at 1x1 layer
        Output:
          lgt_glb_mlp: class logits from global features
          lgt_glb_lin: class logits from global features
        '''
        # collect features to feed into classifiers
        # - always detach() -- send no grad into encoder!
        if not self.is_ft:
            h_top_cls = flatten(ftr_1).detach()
        else:
            h_top_cls = flatten(ftr_1)
        # h_top_cls = flatten(ftr_1)
        # compute predictions
        lgt_glb_mlp = self.block_glb_mlp(h_top_cls)
        # lgt_glb_lin = self.block_glb_lin(h_top_cls)
        return lgt_glb_mlp


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

        init_weights(self, 'kaiming')

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.c0 = nn.Conv2d(in_channel, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 10 * 10 + 128, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        relu = nn.ReLU(True)
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, 512, kernel_size=1),
            relu,
            nn.Conv2d(512, 512, kernel_size=1),
            relu,
            nn.Conv2d(512, 1, kernel_size=1)
        )

        init_weights(self, 'kaiming')

    def forward(self, x):
        return self.model(x)


class GANDiscriminator(nn.Module):
    # initializers
    def __init__(self, cfg, device=None):
        super(GANDiscriminator, self).__init__()
        self.cfg = cfg
        self.device = device
        norm = nn.BatchNorm2d
        self.d_downsample_num = 4

        distribute = [
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            norm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            norm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        ]

        self.criterion = nn.BCELoss() if cfg.NO_LSGAN else nn.MSELoss()
        if self.cfg.NO_LSGAN:
            distribute.append(nn.Sigmoid())

        self.distribute = nn.Sequential(*distribute)

    def forward(self, x, target):
        # distribution
        pred = self.distribute(x)

        if target:
            label = 1
        else:
            label = 0

        dis_patch = torch.FloatTensor(pred.size()).fill_(label).to(self.device)
        loss = self.criterion(pred, dis_patch)

        return loss


class GANDiscriminator_Image(nn.Module):
    # initializers
    def __init__(self, cfg, device=None):
        super(GANDiscriminator_Image, self).__init__()
        self.cfg = cfg
        self.device = device
        norm = nn.BatchNorm2d
        self.d_downsample_num = 4
        relu = nn.LeakyReLU(0.2)

        distribute = [
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            norm(64),
            relu,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            relu,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm(256),
            relu,
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            norm(256),
            relu,
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            norm(512),
            relu,
            nn.Conv2d(512, 1, kernel_size=1),
        ]

        self.criterion = nn.BCELoss() if cfg.NO_LSGAN else nn.MSELoss()
        if self.cfg.NO_LSGAN:
            distribute.append(nn.Sigmoid())

        self.distribute = nn.Sequential(*distribute)
        init_weights(self, 'normal')

    def forward(self, x, target):
        # distribution
        pred = self.distribute(x)

        if target:
            label = 1
        else:
            label = 0

        dis_patch = torch.FloatTensor(pred.size()).fill_(label).to(self.device)
        loss = self.criterion(pred, dis_patch)

        return loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class PriorDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.l0 = nn.Linear(in_channel, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

        init_weights(self, 'normal')

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))



class BasicBlockWithoutNorm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockWithoutNorm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

        if inplanes == planes:
            kernel_size, padding = 3, 1
        else:
            kernel_size, padding = 1, 0

        self.downsample = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False)

        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18], norm=nn.BatchNorm2d):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          norm(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                norm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            norm(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(3, reduction_dim, kernel_size=1, bias=False),
            norm(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out
