import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

sys.path.append('./')
sys.path.append('../')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead

def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride==1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)

class FeatureSelection(nn.Module):
    def __init__(self,
                 channel,
                 reduction=16):
        super(FeatureSelection, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HOTA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(HOTA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
        self.horizontal_conv_weight = nn.Parameter(torch.randn(f, f, 1, 3))
        self.horizontal_conv_bias = nn.Parameter(torch.randn(f,))
        
        self.vertical_conv_weight = nn.Parameter(torch.randn(f, f, 3, 1))
        self.vertical_conv_bias = nn.Parameter(torch.randn(f,))
        
        self.se = FeatureSelection(f*6)
        self.conv_reduce = conv(f*6, f, kernel_size=3, padding=1)

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        res = x
        c1_ = (self.conv1(x))
        
        c1_v1 = F.conv2d(c1_, self.vertical_conv_weight, bias=self.vertical_conv_bias, stride=(2,2), padding=(1,0), dilation=1)
        c1_v2 = F.conv2d(c1_, self.vertical_conv_weight, bias=self.vertical_conv_bias, stride=(2,2), padding=(2,0), dilation=2)
        c1_v3 = F.conv2d(c1_, self.vertical_conv_weight, bias=self.vertical_conv_bias, stride=(2,2), padding=(3,0), dilation=3)
        
        c1_h1 = F.conv2d(c1_, self.horizontal_conv_weight, bias=self.horizontal_conv_bias, stride=(2,2), padding=(0,1), dilation=1)
        c1_h2 = F.conv2d(c1_, self.horizontal_conv_weight, bias=self.horizontal_conv_bias, stride=(2,2), padding=(0,2), dilation=2)
        c1_h3 = F.conv2d(c1_, self.horizontal_conv_weight, bias=self.horizontal_conv_bias, stride=(2,2), padding=(0,3), dilation=3)
        
        c1_fusion = torch.cat([c1_v1, c1_v2, c1_v3, c1_h1, c1_h2, c1_h3], dim=1)
        c1_selected = self.se(c1_fusion)
        c1_selected = self.conv_reduce(c1_selected)
        
        v_max = F.max_pool2d(c1_selected, kernel_size=8, stride=4)
        
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        
        c3 = F.interpolate(c3, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        
        cf = self.conv_f(c1_)
        c4 = self.conv4(cf+c3)
        m = self.sigmoid(c4)
        
        return res * m



class PCAN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32):
        super(PCAN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        # for i in range(srb_nums):
        #     setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        # PSPB
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), PCAB(2*hidden_units, i+2))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units*srb_nums, 2*hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2*hidden_units)
                ))
        
        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [32, 64]
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
        self.spatial_attention = HOTA(n_feats=2*hidden_units*srb_nums)

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        if self.stn and self.training:
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        x = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x, x2)
        x4 = self.block4(x, x2, x3)
        x5 = self.block5(x, x2, x3, x4)
        x6 = self.block6(x, x2, x3, x4, x5)
        x7 = self.block7(self.spatial_attention(torch.cat([x2, x3, x4, x5, x6], 1)))

        output = torch.tanh(self.block8(x7+x))
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
        # print("emb:", t_embedding.shape)
        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)



class PCAN_TL(nn.Module):
    def __init__(self, scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 text_emb=37, #26+26+1
                 out_text_channels=32):
        super(PCAN_TL, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        # for i in range(srb_nums):
        #     setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        # PSPB
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), PCAB_TP(2 * hidden_units, i + 2))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units * srb_nums, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [32, 64]
        tps_outputsize = [height // scale_factor, width // scale_factor]
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
        self.spatial_attention = HOTA(n_feats=2 * hidden_units * srb_nums)

        self.infoGen = InfoGen(text_emb, out_text_channels)

    def forward(self, x, text_emb=None):
        # import ipdb;ipdb.set_trace()
        if self.stn and self.training:
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)

        block_1 = x = self.block1(x)

        if text_emb is None:
            text_emb = torch.zeros(1, 37, 1, 26).to(x.device) #37
        # padding_feature = block['1']

        spatial_t_emb_gt, pr_weights_gt = None, None

        spatial_t_emb, pr_weights = self.infoGen(text_emb)
        spatial_t_emb = F.interpolate(spatial_t_emb, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        x2 = self.block2([x], spatial_t_emb)
        x3 = self.block3([x, x2], spatial_t_emb)
        x4 = self.block4([x, x2, x3], spatial_t_emb)
        x5 = self.block5([x, x2, x3, x4], spatial_t_emb)
        x6 = self.block6([x, x2, x3, x4, x5], spatial_t_emb)
        x7 = self.block7(self.spatial_attention(torch.cat([x2, x3, x4, x5, x6], 1)))

        output = torch.tanh(self.block8(x7 + x))

        if self.training:
            ret_mid = {
                "pr_weights": pr_weights,
                "pr_weights_gt": pr_weights_gt,
                "spatial_t_emb": spatial_t_emb,
                "spatial_t_emb_gt": spatial_t_emb_gt,
                "in_feat": block_1,
                "trans_feat": spatial_t_emb
            }

            return output, ret_mid

        else:
            return output, pr_weights


class PCAB(nn.Module):
    def __init__(self, channels, no):
        super(PCAB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = mish()

        self.conv2_w = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2_w = nn.BatchNorm2d(channels)
        self.conv2_h = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2_h = nn.BatchNorm2d(channels)
        
        self.gru1 = GruBlock(channels, channels)
        self.gru2 = GruBlock(channels, channels)
        reduction = 16
        self.fs = FeatureSelection(channels*2, reduction)
        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.prelu3 = mish()
        self.conv_reduce = nn.Conv2d(channels*(no-1), channels, kernel_size=1, padding=0)
    def forward(self, *x):
        x = torch.cat(x, 1)
        x = self.conv_reduce(x)
        
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        
        residual_w = self.conv2_w(residual)
        residual_w = self.bn2_w(residual_w)
        w_feat = self.gru1(x+residual_w)
        
        residual_h = self.conv2_h(residual)
        residual_h = self.bn2_h(residual_h)
        h_feat = self.gru2((x+residual_h).transpose(-1, -2)).transpose(-1, -2)
        
        fusion_feat = self.fs(torch.cat([h_feat, w_feat], dim=1))

        return self.prelu3(self.bn3(self.conv3(fusion_feat)))


class PCAB_TP(nn.Module):
    def __init__(self, channels, no):
        super(PCAB_TP, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = mish()

        self.conv2_w = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2_w = nn.BatchNorm2d(channels)
        self.conv2_h = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2_h = nn.BatchNorm2d(channels)

        self.gru1 = GruBlock(channels + 32, channels)
        self.gru2 = GruBlock(channels + 32, channels)
        reduction = 16
        self.fs = FeatureSelection(channels * 2, reduction)
        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.prelu3 = mish()
        self.conv_reduce = nn.Conv2d(channels * (no - 1), channels, kernel_size=1, padding=0)

    def forward(self, x, tp):
        x = torch.cat(x, 1)
        x = self.conv_reduce(x)

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)

        residual_w = self.conv2_w(residual)
        residual_w = self.bn2_w(residual_w)
        w_feat = self.gru1(torch.cat([x + residual_w, tp], 1))

        residual_h = self.conv2_h(residual)
        residual_h = self.bn2_h(residual_h)
        h_feat = self.gru2(torch.cat([x + residual_w, tp], 1).transpose(-1, -2)).transpose(-1, -2)

        fusion_feat = self.fs(torch.cat([h_feat, w_feat], dim=1))

        return self.prelu3(self.bn3(self.conv3(fusion_feat)))


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x



if __name__ == '__main__':
    # net = NonLocalBlock2D(in_channels=32)
    img = torch.zeros(7, 3, 16, 64)
    embed()
