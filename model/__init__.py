import torch
import torch.nn.functional as F

def torch_distortion(torch_image_batches, arc_batches, rand_offs, off_range=0.2):
    # ratios: H / W

    device = torch_image_batches.device

    N, C, H, W = torch_image_batches.shape
    ratios = H / float(W)

    # rand_offs = random.random() * (1 - ratios)
    ratios_mul = ratios + (rand_offs.unsqueeze(1) * off_range * 2) - off_range

    a11, a12, a21, a22 = torch.cos(arc_batches), \
                         torch.sin(arc_batches), \
                         -torch.sin(arc_batches), \
                         torch.cos(arc_batches)

    x_shift = torch.zeros_like(arc_batches)
    y_shift = torch.zeros_like(arc_batches)

    affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1) * ratios_mul, x_shift.unsqueeze(1),
                               a21.unsqueeze(1) / ratios_mul, a22.unsqueeze(1), y_shift.unsqueeze(1)], dim=1)
    affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

    affine_grid = F.affine_grid(affine_matrix, torch_image_batches.shape)
    distorted_batches = F.grid_sample(torch_image_batches, affine_grid)

    return distorted_batches