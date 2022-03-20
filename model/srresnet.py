import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
sys.path.append('./')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead
from IPython import embed
from .transformer_v2 import InfoTransformer
from .transformer_v2 import PositionalEncoding

class SRResNet(nn.Module):
    def __init__(self, scale_factor=2, STN=False, width=128, height=32, mask=False):
        upsample_block_num = int(math.log(scale_factor, 2))
        super(SRResNet, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, in_planes, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        # embed()
        #if self.stn and self.training:
        #    _, ctrl_points_x = self.stn_head(x)
        #    x, _ = self.tps(x, ctrl_points_x)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        self.block = block7

        return F.tanh(block8)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class InfoGenTrans(nn.Module):
    def __init__(
                self,
                t_emb,
                out_text_channels,
                output_size=(16, 64),
                # d_model=512,
                t_encoder_num=1,
                t_decoder_num=2,
                 ):
        super(InfoGenTrans, self).__init__()

        d_model = out_text_channels # * output_size[0]

        self.fc_in = nn.Linear(t_emb, d_model)
        self.activation = nn.PReLU()

        self.upsample_transformer = InfoTransformer(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=64,
                                          num_encoder_layers=t_encoder_num,
                                          num_decoder_layers=t_decoder_num,
                                          normalize_before=False,
                                          return_intermediate_dec=True, )

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0]
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.dropout = nn.Dropout(0.3)

        self.masking = torch.ones(output_size)

        # self.tp_uper = InfoGen(t_emb, out_text_channels)

    def forward(self, image_feature, tp_input):

        H, W = self.output_size
        x = tp_input

        # x_tar = tp_input
        # x_tar = self.tp_uper(x_tar)
        N_i, C_i, H_i, W_i = image_feature.shape

        x_tar = image_feature #F.interpolate(x_tar, (H_i, W_i), mode='bilinear', align_corners=True)
        # x_im = None
        # x_im = x_tar.view(N_i, C_i * H_i, W_i).permute(2, 0, 1)
        # [1024, N, 64]
        x_im = x_tar.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)
        # print("masking1:", self.masking.shape, self.masking)

        # if self.training:
        #     # x_im: [N, C, H, W]
        #     masking = self.dropout(self.masking).unsqueeze(0).unsqueeze(0)
        # else:
        #     masking = self.masking

        # print("masking:", image_feature.shape)

        device = x.device
        # print('x:', x.shape)
        x = x.permute(0, 3, 1, 2).squeeze(-1)
        x = self.activation(self.fc_in(x))
        N, L, C = x.shape

        x_pos = self.pe(torch.zeros((N, L, C)).to(device)).permute(1, 0, 2)
        mask = torch.zeros((N, L)).to(device).bool()
        x = x.permute(1, 0, 2)

        # print("x_im:", x_im.shape)

        text_prior, pr_weights = self.upsample_transformer(x, mask, self.init_factor.weight, x_pos, tgt=x_im)
        text_prior = text_prior.mean(0)

        # [N, L, C] -> [N, C, H, W]

        text_prior = text_prior.permute(1, 2, 0).view(N, C, H, W) #// H

        return text_prior, pr_weights



class SRResNet_TL(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 STN=False,
                 width=128,
                 height=32,
                 mask=False,
                 text_emb=37,  # 26+26+1
                 out_text_channels=64
                 ):

        upsample_block_num = int(math.log(scale_factor, 2))
        super(SRResNet_TL, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock_TL(64, out_text_channels)
        self.block3 = ResidualBlock_TL(64, out_text_channels)
        self.block4 = ResidualBlock_TL(64, out_text_channels)
        self.block5 = ResidualBlock_TL(64, out_text_channels)
        self.block6 = ResidualBlock_TL(64, out_text_channels)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, in_planes, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        self.tps_outputsize = [height//scale_factor, width//scale_factor]
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
        self.infoGen = InfoGenTrans(text_emb, out_text_channels)
        # InfoGen(text_emb, out_text_channels)

    def forward(self, x, text_emb=None):
        # embed()

        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block1 = self.block1(x)

        spatial_t_emb, pr_weights = self.infoGen(block1, text_emb)
        spatial_t_emb = F.interpolate(spatial_t_emb, self.tps_outputsize, mode='bilinear', align_corners=True)

        block2 = self.block2(block1, spatial_t_emb)
        block3 = self.block3(block2, spatial_t_emb)
        block4 = self.block4(block3, spatial_t_emb)
        block5 = self.block5(block4, spatial_t_emb)
        block6 = self.block6(block5, spatial_t_emb)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return F.tanh(block8), pr_weights


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


class ResidualBlock_TL(nn.Module):
    def __init__(self, channels, out_text_channels=32):
        super(ResidualBlock_TL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels+out_text_channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, text_emb):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)

        ############ Fusing with TL ############
        cat_feature = torch.cat([residual, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################

        residual = self.conv2(cat_feature)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))
