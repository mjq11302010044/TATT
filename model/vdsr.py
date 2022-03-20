import torch
import torch.nn as nn
from math import sqrt
from IPython import embed
import sys
sys.path.append('./')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead
import torch.nn.functional as F

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x)) + x


class Conv_ReLU_Block_TL(nn.Module):
    def __init__(self, out_text_channels=32):
        super(Conv_ReLU_Block_TL, self).__init__()
        self.conv = nn.Conv2d(in_channels=64 + out_text_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, text_emb):
        # x, text_emb = x_in
        # print("x.shape", type(x_in))
        ############ Fusing with TL ############
        cat_feature = torch.cat([x, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################

        return self.relu(self.conv(cat_feature)) + x



class VDSR(nn.Module):
    def __init__(self, scale_factor=2, in_planes=3, width=32, height=128, STN=False):
        super(VDSR, self).__init__()
        self.upscale_factor = scale_factor
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 6)
        self.input = nn.Conv2d(in_channels=in_planes, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height, width]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = False
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=3,
                num_ctrlpoints=num_control_points,
                activation='none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.stn:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upscale_factor)
        residual = x
        out = self.relu(self.input(x))
        # out = self.residual_layer(out)

        for block in self.residual_layer:
            out = block(out)

        out = self.output(out)
        out = torch.add(out, residual)
        return out


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


class VDSR_TL(nn.Module):
    def __init__(self, scale_factor=2,
                 in_planes=4,
                 width=32,
                 height=128,
                 STN=False,
                 text_emb=37,  # 26+26+1
                 out_text_channels=32):

        # print("in_planes:", in_planes)

        super(VDSR_TL, self).__init__()
        self.upscale_factor = scale_factor
        self.out_text_channels = out_text_channels
        # self.residual_layer = self.make_layer(Conv_ReLU_Block_TL, 8)
        self.input = nn.Conv2d(in_channels=in_planes, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # From [1, 1] -> [16, 16]
        self.infoGen = InfoGen(text_emb, out_text_channels)

        block = Conv_ReLU_Block_TL

        self.block1 = block(self.out_text_channels)
        self.block2 = block(self.out_text_channels)
        self.block3 = block(self.out_text_channels)
        self.block4 = block(self.out_text_channels)
        self.block5 = block(self.out_text_channels)
        self.block6 = block(self.out_text_channels)
        #self.block7 = block(self.out_text_channels)
        #self.block8 = block(self.out_text_channels)
        #self.block9 = block(self.out_text_channels)
        #self.block10 = block(self.out_text_channels)
        #self.block11 = block(self.out_text_channels)
        #self.block12 = block(self.out_text_channels)
        #self.block13 = block(self.out_text_channels)
        #self.block14 = block(self.out_text_channels)
        #self.block15 = block(self.out_text_channels)
        #self.block16 = block(self.out_text_channels)
        #self.block17 = block(self.out_text_channels)
        #self.block18 = block(self.out_text_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        self.tps_outputsize = [height, width]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = False
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.out_text_channels))

        # self.internal_layers = nn.Sequential(*layers)

        return nn.Sequential(*layers)

    def forward(self, x, text_emb=None):
        if self.stn:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upscale_factor)

        spatial_t_emb = self.infoGen(text_emb)
        spatial_t_emb = F.interpolate(spatial_t_emb, self.tps_outputsize, mode='bilinear', align_corners=True)

        residual = x
        out = self.relu(self.input(x))

        # for block in self.blocks:
        # out = self.block1(out, spatial_t_emb)
        out = self.block1(out, spatial_t_emb)
        out = self.block2(out, spatial_t_emb)
        out = self.block3(out, spatial_t_emb)
        out = self.block4(out, spatial_t_emb)
        out = self.block5(out, spatial_t_emb)
        out = self.block6(out, spatial_t_emb)
        #out = self.block7(out, spatial_t_emb)
        #out = self.block8(out, spatial_t_emb)
        #out = self.block9(out, spatial_t_emb)
        #out = self.block10(out, spatial_t_emb)

        # out, spatial_t_emb = out

        # out = self.block1(out, spatial_t_emb)

        out = self.output(out)
        out = torch.add(out, residual)
        return out


if __name__=='__main__':
    embed()
