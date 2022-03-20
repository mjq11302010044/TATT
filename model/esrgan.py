import functools
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from IPython import embed


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDB_TL(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, text_channel=32):
        super(RRDB_TL, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

        self.proj = nn.Conv2d(nf + text_channel, nf, 1, 1, 0)
        self.activation = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        x, text_emb = x

        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        im_feat = out * 0.2 + x
        cat_feat = self.bn2(self.proj(torch.cat([im_feat, text_emb], 1)))

        return [cat_feat + im_feat, text_emb]


class RRDBNet(nn.Module):
    def __init__(self, scale_factor=2, in_nc=4, out_nc=4, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.upsample_block_num = int(math.log(scale_factor, 2))
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        # self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        for i in range(self.upsample_block_num):
            setattr(self, 'upconv%d' % (i + 1),
                    nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                    )
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        for i in range(self.upsample_block_num):
            fea = self.lrelu(getattr(self, 'upconv%d' % (i + 1))(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

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
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)


class RRDBNet_TL(nn.Module):
    def __init__(self, scale_factor=2, in_nc=4, out_nc=4, nf=64, nb=23, gc=32, text_emb=37, out_text_channels=32):
        super(RRDBNet_TL, self).__init__()
        RRDB_block_f = functools.partial(RRDB_TL, nf=nf, gc=gc, text_channel=out_text_channels)
        self.upsample_block_num = int(math.log(scale_factor, 2))
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        # self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        for i in range(self.upsample_block_num):
            setattr(self, 'upconv%d' % (i + 1),
                    nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                    )
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.infoGen = InfoGen(text_emb, out_text_channels)

    def forward(self, x, text_emb):
        fea = self.conv_first(x)

        # all_pred_vecs = []
        spatial_t_emb_gt, pr_weights_gt = None, None
        spatial_t_emb_, pr_weights = self.infoGen(text_emb)  # # block['1'],block['1'],
        spatial_t_emb = F.interpolate(spatial_t_emb_, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        trunk, _ = self.RRDB_trunk([fea, spatial_t_emb])

        trunk = self.trunk_conv(trunk)
        fea = fea + trunk
        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        for i in range(self.upsample_block_num):
            fea = self.lrelu(getattr(self, 'upconv%d' % (i + 1))(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out, pr_weights



def conv_block(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
               act_type='leakyrelu', pad_type='reflection', norm_type=None, negative_slope=0.2, n_prelu=1,
               inplace=True, n_padding=None):
    n_pad = n_padding if n_padding else get_n_padding(kernel_size, dilation)
    pad = padding(pad_type, n_pad) if pad_type else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
    norm = normalization(norm_type, out_channels) if norm_type else None
    act = activation(act_type, inplace=inplace, negative_slope=negative_slope, n_prelu=n_prelu) if act_type else None
    if (norm is None) and (act_type is None):
        return nn.Sequential(pad, conv)
    if pad_type is None:
        return nn.Sequential(conv, act)
    if norm is None:
        return nn.Sequential(pad, conv, act)
    else:
        return nn.Sequential(pad, conv, norm, act)


class SubDiscriminator(nn.Module):
    def __init__(self, act_type='leakyrelu', num_conv_block=4):
        super(SubDiscriminator, self).__init__()

        block = []

        in_channels = 3
        out_channels = 64

        for _ in range(num_conv_block):
            block += conv_block(in_channels, out_channels, stride=1, act_type=act_type, pad_type=None,
                                norm_type='instancenorm')
            in_channels = out_channels
            block += conv_block(in_channels, out_channels, stride=2, act_type=act_type, n_padding=1)
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [nn.Conv2d(in_channels, out_channels, 3),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(out_channels, out_channels, 3)]

        self.feature_extraction = nn.Sequential(*block)

        self.classification = nn.Sequential(
            nn.Linear(512 * 9 * 9, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


if __name__ == '__main__':
    embed()