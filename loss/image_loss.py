import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from IPython import embed
from torchvision import transforms
from torch.autograd import Variable

class ImageLoss(nn.Module):
    def __init__(self, gradient=True, loss_weight=[20, 1e-4]):
        super(ImageLoss, self).__init__()
        self.mse = nn.MSELoss(reduce=False)
        if gradient:
            self.GPLoss = GradientPriorLoss()
        self.gradient = gradient
        self.loss_weight = loss_weight

    def forward(self, out_images, target_images, grad_mask=None):

        # if not grad_mask is None:
        #     out_images *= grad_mask
        #     target_images *= grad_mask

        if self.gradient:

            mse_loss = self.mse(out_images, target_images).mean(1).mean(1).mean(1)

            loss = self.loss_weight[0] * mse_loss + \
                   self.loss_weight[1] * self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :])
        else:
            loss = self.loss_weight[0] * mse_loss

        return loss


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss(reduce=False)

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)

        g_loss = self.func(map_out, map_target)

        return g_loss.mean(1).mean(1).mean(1)

    @staticmethod
    def gradient_map(x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad


class EdgeImageLoss(nn.Module):
    def __init__(self, gradient=True, edge=False, loss_weight=[20, 1e-4, 1e-4]):
        super(EdgeImageLoss, self).__init__()
        self.mse = nn.MSELoss()

        self.EGLoss = EdgeGuidanceLoss()
        self.gradient = gradient
        self.edge = edge
        self.loss_weight = loss_weight

    def forward(self, out_images, target_images, grad_mask=None):

        # if not grad_mask is None:
        #     out_images *= grad_mask
        #     target_images *= grad_mask

        loss = self.loss_weight[0] * self.mse(out_images, target_images)
        loss += self.loss_weight[2] * self.EGLoss(out_images[:, :3, :, :], target_images[:, :3, :, :]) * 0.1

        return loss


class EdgeGuidanceLoss(nn.Module):
    def __init__(self, ):
        super(EdgeGuidanceLoss, self).__init__()
        self.func = nn.L1Loss()

        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.cuda.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)
        self.weight_hori = self.weight_const_hori

        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.cuda.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)
        self.weight_vertical = self.weight_const_vertical

    def forward(self, out_images, target_images):

        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target)

    def gradient_map(self, x):
        weight_hori = Variable(self.weight_hori)
        weight_vertical = Variable(self.weight_vertical)

        weight_hori = weight_hori.expand(3, 3, 3, 3)
        weight_vertical = weight_vertical.expand(3, 3, 3, 3)

        weight_hori.requires_grad = False
        weight_vertical.requires_grad = False

        try:
            x_hori = F.conv2d(x, weight_hori, padding=1)
            # x_hori = self.conv2d_hori(x3)
        except:
            print('horizon error')
        try:
            x_vertical = F.conv2d(x, weight_vertical, padding=1)
            # x_vertical = self.conv2d_vertical(x3)
        except:
            print('vertical error')

        edge_detect = torch.pow(torch.pow(x_hori * 0.5, 2) + torch.pow(x_vertical * 0.5, 2) + 1e-6, 0.5)
        return edge_detect

if __name__ == '__main__':
    im1=Image.open('../tt.jpg')
    im1=transforms.ToTensor()(im1)
    im1=im1.unsqueeze(0)
    im2 = Image.open('../tt1.jpg')
    im2 = transforms.ToTensor()(im2)
    im2 = im2.unsqueeze(0)
    embed()
