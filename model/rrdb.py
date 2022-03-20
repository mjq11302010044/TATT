import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


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


class ResidualDenseBlock_5C_TL(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, out_text_channels=32):
        super(ResidualDenseBlock_5C_TL, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc + out_text_channels, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x, text_emb):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4, text_emb), 1))

        return x5 * 0.166 + x


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

    def __init__(self, nf, gc=32, out_text_channels=32):
        super(RRDB_TL, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_TL(nf, gc, out_text_channels)
        self.RDB2 = ResidualDenseBlock_5C_TL(nf, gc, out_text_channels)
        self.RDB3 = ResidualDenseBlock_5C_TL(nf, gc, out_text_channels)

    def forward(self, x_in):

        (x, text_emb) = x_in

        out = self.RDB1(x, text_emb)
        out = self.RDB2(out, text_emb)
        out = self.RDB3(out, text_emb)
        return out * 0.2 + x



class RRDBNet_TL(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, text_emb=37,  # 26+26+1
                 out_text_channels=32):
        super(RRDBNet_TL, self).__init__()
        RRDB_block_f = functools.partial(RRDB_TL, nf=nf, gc=gc, out_text_channels=out_text_channels)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, text_emb=None):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk((fea, text_emb)))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out



class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


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


class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.discriminator_a = SubDiscriminator()
        self.discriminator_b = SubDiscriminator()
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, b):
        a = self.discriminator_a(a)
        b = self.discriminator_b(b)
        return self.sigmoid(a - b)


if __name__ == '__main__':
    embed()