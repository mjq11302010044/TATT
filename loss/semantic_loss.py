import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from IPython import embed
from torchvision import transforms
from torch.autograd import Variable

class SemanticLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SemanticLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.margin = margin

        self.lambda1 = 1.0
        self.lambda2 = 1.0

        self.kl_loss = torch.nn.KLDivLoss()

    def forward(self, pred_vec, gt_vec):
        # pred_vec: [N, C]
        # gt_vec: [N, C]
        # mean_sim = torch.mean(self.cos_sim(gt_vec, pred_vec))
        # sim_loss = 1 - mean_sim
        
        #noise =  Variable(torch.rand(pred_vec.shape)) * 0.1 - 0.05

        #normed_pred_vec = pred_vec + noise.to(pred_vec.device)
        # print("pred_vec:", pred_vec.shape)
        norm_vec = torch.abs(gt_vec - pred_vec)
        margin_loss = torch.mean(norm_vec) #

        # pr int("sem_loss:", float(margin_loss.data), "sim_loss:", float(sim_loss.data))
        ce_loss = self.kl_loss(torch.log(pred_vec + 1e-20), gt_vec + 1e-20)
        # print("sem_loss:", float(margin_loss.data), "sim_loss:", float(sim_loss.data))

        return self.lambda1 * margin_loss + self.lambda2 * ce_loss# ce_loss #margin_loss # + ce_loss #  + sim_loss #margin_loss +

    def cross_entropy(self, pred_vec, gt_vec, l=1e-5):
        cal = gt_vec * torch.log(pred_vec+l) + (1 - gt_vec) * torch.log(1 - pred_vec+l)
        #print("cal:", cal)
        return -cal


if __name__ == '__main__':
    im1=Image.open('../tt.jpg')
    im1=transforms.ToTensor()(im1)
    im1=im1.unsqueeze(0)
    im2 = Image.open('../tt1.jpg')
    im2 = transforms.ToTensor()(im2)
    im2 = im2.unsqueeze(0)
    embed()
