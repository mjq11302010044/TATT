import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from IPython import embed


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, nChannel=3, nDenselayer=6, nFeat=64, scale_factor=2, growthRate=32):
        super(RDN, self).__init__()

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat * scale_factor * scale_factor, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale_factor)
        # conv
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        F_ = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)

        output = self.conv3(us)

        return output


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

        return x


# Residual dense block (RDB) architecture
class RDB_TL(nn.Module):
    def __init__(
                self,
                 nChannels,
                 nDenselayer,
                 growthRate,
                 out_text_channels=32
                 ):
        super(RDB_TL, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_+out_text_channels, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x, text_emb):
        out = self.dense_layers(x)

        ############ Fusing with TL ############
        cat_feature = torch.cat([out, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################

        out = self.conv_1x1(cat_feature)
        out = out + x
        return out



# Residual Dense Network
class RDN_TL(nn.Module):
    def __init__(self,
                 nChannel=4,
                 nDenselayer=6,
                 nFeat=64,
                 scale_factor=2,
                 growthRate=32,
                 output_size=(32, 128),
                 text_emb=37,  # 26+26+1
                 out_text_channels=32,
                 ):
        super(RDN_TL, self).__init__()

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3
        self.RDB1 = RDB_TL(nFeat, nDenselayer, growthRate, out_text_channels)
        self.RDB2 = RDB_TL(nFeat, nDenselayer, growthRate, out_text_channels)
        self.RDB3 = RDB_TL(nFeat, nDenselayer, growthRate, out_text_channels)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat * scale_factor * scale_factor, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale_factor)
        # conv
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)

        self.tps_outputsize = [8, 32]

        # From [1, 1] -> [16, 16]
        self.infoGen = InfoGen(text_emb, out_text_channels)

    def forward(self, x, text_emb=None):

        spatial_t_emb = self.infoGen(text_emb)
        spatial_t_emb = F.interpolate(spatial_t_emb, self.tps_outputsize, mode='bilinear', align_corners=True)

        F_ = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0, spatial_t_emb)
        F_2 = self.RDB2(F_1, spatial_t_emb)
        F_3 = self.RDB3(F_2, spatial_t_emb)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)

        output = self.conv3(us)

        return output


if __name__=='__main__':
    embed()