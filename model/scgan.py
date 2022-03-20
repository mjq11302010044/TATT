import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as d_sets
from torch.utils.data import DataLoader as d_loader
import matplotlib.pyplot as plt
from PIL import Image
from IPython import embed
import sys
sys.path.append('./')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead

class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)


class SCGAN_TL(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 in_planes=4,
                 STN=False,
                 height=32,
                 width=128,
                 text_emb=37,  # 26+26+1
                 out_text_channels=32
                 ):
        super(SCGAN_TL, self).__init__()
        self.upscale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()

        self.conva = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relua = nn.ReLU()
        self.convb = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relub = nn.ReLU()
        self.convc = nn.Conv2d(64 + out_text_channels, 64, kernel_size=5, padding=2)
        self.reluc = nn.ReLU()
        self.convd = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relud = nn.ReLU()
        self.conve = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relue = nn.ReLU()
        self.convf = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.reluf = nn.ReLU()
        self.convg = nn.Conv2d(64 + out_text_channels, 64, kernel_size=5, padding=2)
        self.relug = nn.ReLU()
        self.convh = nn.Conv2d(64, in_planes, kernel_size=1, padding=0)
        self.reluh = nn.ReLU()

        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        self.tps_outputsize = [height, width]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(self.tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

        # From [1, 1] -> [16, 16]
        self.infoGen = InfoGen(text_emb, out_text_channels)

    def forward(self, x, text_emb=None):

        # print("x", x.shape, text_emb.shape)

        if self.stn:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upscale_factor)

        spatial_t_emb_, pr_weights = self.infoGen(text_emb)  # # block['1'],block['1'],
        spatial_t_emb = F.interpolate(spatial_t_emb_, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        out = self.conv1(x)
        out = self.relu1(out)

        out = self.relua(self.conva(out))
        out = self.relub(self.convb(out))
        out = torch.cat([out, spatial_t_emb], 1)
        out = self.reluc(self.convc(out))
        out = self.relud(self.convd(out))
        out = self.relue(self.conve(out))
        out = self.reluf(self.convf(out))
        out = torch.cat([out, spatial_t_emb], 1)
        out = self.relug(self.convg(out))
        out = self.reluh(self.convh(out))

        return out, pr_weights


class SCGAN(nn.Module):
    def __init__(self, scale_factor=2, in_planes=4, STN=False, height=32, width=128):
        super(SCGAN, self).__init__()
        self.upscale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()

        self.conva = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relua = nn.ReLU()
        self.convb = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relub = nn.ReLU()
        self.convc = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.reluc = nn.ReLU()
        self.convd = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relud = nn.ReLU()
        self.conve = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relue = nn.ReLU()
        self.convf = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.reluf = nn.ReLU()
        self.convg = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.relug = nn.ReLU()
        self.convh = nn.Conv2d(64, in_planes, kernel_size=1, padding=0)
        self.reluh = nn.ReLU()

        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height, width]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=3,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)

        if self.stn:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        else:
            out = torch.nn.functional.interpolate(out, scale_factor=self.upscale_factor)

        out = self.relua(self.conva(out))
        out = self.relub(self.convb(out))
        out = self.reluc(self.convc(out))
        out = self.relud(self.convd(out))
        out = self.relue(self.conve(out))
        out = self.reluf(self.convf(out))
        out = self.relug(self.convg(out))
        out = self.reluh(self.convh(out))

        return out