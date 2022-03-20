from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from IPython import embed


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 1]

    mse = ((img1[:,:3,:,:]*255 - img2[:,:3,:,:]*255)**2).mean()
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def weighted_calculate_psnr(img1, img2, weighted_mask):
    # img1 and img2 have range [0, 1]
    # print("weighted_mask:", weighted_mask.shape, img1[:,:3,:,:].shape)
    mse = ((img1[:,:3,:,:] * weighted_mask * 255 - img2[:,:3,:,:] * weighted_mask*255)**2)
    mse = mse.mean()
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_rect_window(window_H, window_W, channel):
    _1D_window_H = gaussian(window_H, 1.5).unsqueeze(1)
    _1D_window_W = gaussian(window_W, 1.5).unsqueeze(1)

    _2D_window = _1D_window_H.mm(_1D_window_W.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_H, window_W).contiguous())
    return window


def _ssim_weighted(img1_, img2_, window, window_size, channel, weighted_mask, size_average=True):

    img1 = img1_ * weighted_mask
    img2 = img2_ * weighted_mask

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _tri_ssim(img1, img2, img3, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu3 = F.conv2d(img3, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu3_sq = mu3.pow(2)
    mu1_mu2 = mu1 * mu2
    mu2_mu3 = mu2 * mu3
    mu3_mu1 = mu3 * mu1

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma3_sq = F.conv2d(img3 * img3, window, padding=window_size // 2, groups=channel) - mu3_sq

    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma23 = F.conv2d(img2 * img3, window, padding=window_size // 2, groups=channel) - mu2_mu3
    sigma31 = F.conv2d(img3 * img1, window, padding=window_size // 2, groups=channel) - mu3_mu1

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((mu1_mu2 + mu2_mu3 + mu3_mu1 + C1) * (sigma12 + sigma23 + sigma31 + C2)) \
               / ((mu1_sq + mu2_sq + mu3_sq + C1) * (sigma1_sq + sigma2_sq + sigma3_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_rect(img1, img2, window, window_size, channel, size_average=True):

    H, W = window_size
    padding = (H//2, W//2)

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class Distorted_SSIM(torch.nn.Module):
    def __init__(self, size_average=True):
        super(Distorted_SSIM, self).__init__()
        self.window_sizes = [(5, 11), (11, 5), (11, 11)]

        self.channel = 1

        self.windows = [
                            create_rect_window(self.window_sizes[i][0], self.window_sizes[i][1], self.channel)
                            for i in range(len(self.window_sizes))
                        ]

        self.size_average = size_average

    def forward(self, img1, img2):
        img1 = img1[:,:3,:,:]
        img2 = img2[:,:3,:,:]
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.windows[0].data.type() == img1.data.type():
            windows = self.windows
        else:
            windows = [
                create_rect_window(self.window_sizes[i][0], self.window_sizes[i][1], channel)
                for i in range(len(self.window_sizes))
            ]

            if img1.is_cuda:
                windows = [window.cuda(img1.get_device()) for window in windows]
            # window = window.type_as(img1)
            windows = [window.type_as(img1) for window in windows]

            self.windows = windows
            self.channel = channel

        ssim_each = 0.
        for win_idx in range(len(self.windows)):
            ssim_each += _ssim_rect(img1, img2, windows[win_idx], self.window_sizes[win_idx], channel, self.size_average)

        return ssim_each / len(self.windows)




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        img1 = img1[:,:3,:,:]
        img2 = img2[:,:3,:,:]
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)



class TRI_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(TRI_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, img3):
        # img1 = img1[:,:3,:,:]
        # img2 = img2[:,:3,:,:]
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _tri_ssim(img1, img2, img3, window, self.window_size, channel, self.size_average)


class SSIM_WEIGHTED(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_WEIGHTED, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, weighted_mask):
        img1 = img1[:,:3,:,:]
        img2 = img2[:,:3,:,:]
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_weighted(img1, img2, window, self.window_size, channel, weighted_mask, self.size_average)


class SSIM_TSR(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_TSR, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # img1 = img1[:,:3,:,:]
        # img2 = img2[:,:3,:,:]
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)



def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim_weighted(img1, img2, weighted_mask, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_weighted(img1, img2, window, window_size, channel, weighted_mask, size_average)


if __name__=='__main__':
    embed()