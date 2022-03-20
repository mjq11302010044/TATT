import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import pickle
import copy
from utils import util, ssim_psnr
from IPython import embed
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from PIL import Image
import numpy as np
import cv2
sys.path.append('../')
sys.path.append('./')
from interfaces import base
from utils.meters import AverageMeter
from utils.metrics import get_string_aster, get_string_crnn, Accuracy
from utils.util import str_filt
from utils import utils_moran

from model import gumbel_softmax
from loss.semantic_loss import SemanticLoss
from copy import deepcopy
from tensorboardX import SummaryWriter

import random
import math
import numpy as np
from ptflops import get_model_complexity_info
import editdistance
import time

import lpips

lpips_vgg = lpips.LPIPS(net="vgg")

vis = False
vis_feature = False
# torch.backends.cudnn.enabled = False

TEST_MODEL = "MORAN"
sem_loss = SemanticLoss()
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')

ssim = ssim_psnr.SSIM()

distorted_ssim = ssim_psnr.Distorted_SSIM()

tri_ssim = ssim_psnr.TRI_SSIM()

ABLATION_SET = ["tsrn_tl_cascade_sft", "tsrn_tl_cascade", "srcnn_tl",
                "srresnet_tl", "rdn_tl", "vdsr_tl", "tranSR_v4",
                "esrgan_tl", "scgan_tl", "tbsrn_tl", "tatt", "pcan_tl"]

_DEBUG = False

class TextSR(base.TextBase):

    def SR_confence(self, image, angle):
        pass

    def rotate_img(self, image, angle):
        # convert to cv2 image

        if not angle == 0.0:
            (h, w) = image.shape[:2]
            scale = 1.0
            # set the rotation center
            center = (w / 2, h / 2)
            # anti-clockwise angle in the function
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, M, (w, h))
            # back to PIL image
        return image

    def loss_stablizing(self, loss_set, keep_proportion=0.7):

        # acsending
        sorted_val, sorted_ind = torch.sort(loss_set)
        batch_size = loss_set.shape[0]

        # print("batch_size:", loss_set, batch_size)
        loss_set[sorted_ind[int(keep_proportion * batch_size)]:] = 0.0

        return loss_set


    def cal_all_models(self, model_list, recognizer_list):

        macs = 0.
        params = 0.

        for model in model_list:
            mac, param = get_model_complexity_info(model, (4, 16, 64), as_strings=True,
                                                     print_per_layer_stat=False, verbose=True)

            print('model {:<30}  {:<8}'.format('Computational complexity: ', mac))
            print('model {:<30}  {:<8}'.format('Number of parameters: ', param))

            # macs += mac
            # params += param

        for recognizer in recognizer_list:
            mac, param = get_model_complexity_info(recognizer, (1, 32, 100), as_strings=True,
                                                     print_per_layer_stat=False, verbose=True)

            print('recognizer {:<30}  {:<8}'.format('Computational complexity: ', mac))
            print('recognizer {:<30}  {:<8}'.format('Number of parameters: ', param))

            # macs += mac
            # params += param

        print('{:<30}  {:<8}'.format('Total computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Total number of parameters: ', params))



    def torch_rotate_img(self, torch_image_batches, arc_batches, rand_offs, off_range=0.2):

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

        # print("rand_offs:", rand_offs.shape, a12.shape)

        x_shift = torch.zeros_like(arc_batches)
        y_shift = torch.zeros_like(arc_batches)

        # print("device:", device)
        affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1) * ratios_mul, x_shift.unsqueeze(1),
                                   a21.unsqueeze(1) / ratios_mul, a22.unsqueeze(1), y_shift.unsqueeze(1)], dim=1)
        affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

        affine_grid = F.affine_grid(affine_matrix, torch_image_batches.shape)
        distorted_batches = F.grid_sample(torch_image_batches, affine_grid)

        return distorted_batches


    def yuv_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        r"""Convert an YUV image to RGB.

        The image data is assumed to be in the range of (0, 1).

        Args:
            image (torch.Tensor): YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

        Returns:
            torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

        Example:
            >>> input = torch.rand(2, 3, 4, 5)
            >>> output = yuv_to_rgb(input)  # 2x3x4x5
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                             .format(image.shape))

        y: torch.Tensor = image[..., 0, :, :]
        u: torch.Tensor = image[..., 1, :, :]
        v: torch.Tensor = image[..., 2, :, :]

        r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
        g: torch.Tensor = y + -0.396 * u - 0.581 * v
        b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0

        out: torch.Tensor = torch.stack([r, g, b], -3)

        return out

    def yuv_to_rgb_cv(self, image: torch.Tensor) -> torch.Tensor:

        im_device = image.device
        image_np = image.data.cpu().numpy()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_YUV2RGB)

        return torch.tensor(image_np).to(im_device)


    def rgb_to_yuv(self, image: torch.Tensor) -> torch.Tensor:
        r"""Convert an RGB image to YUV.

        The image data is assumed to be in the range of (0, 1).

        Args:
            image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

        Returns:
            torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

        Example:
            >>> input = torch.rand(2, 3, 4, 5)
            >>> output = rgb_to_yuv(input)  # 2x3x4x5
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                             .format(image.shape))

        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
        u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
        v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

        out: torch.Tensor = torch.stack([y, u, v], -3)

        return out


    def model_inference(self, images_lr, images_hr, model_list, aster, i):

        ret_dict = {}
        ret_dict["label_vecs"] = None

        ret_dict["duration"] = 0

        if self.args.arch == "tsrn":

            before = time.time()
            images_sr = model_list[0](images_lr)
            after = time.time()

            ret_dict["duration"] += (after - before)

            if vis_feature:
                # [N, C, H, W] -> [N, C, W]
                block_feature = model_list[-1].block["7"].mean(2)


        elif self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"]:

            ###############################################
            aster_dict_hr = self.parse_crnn_data(images_lr[:, :3, :, :])
            label_vecs = aster[1](aster_dict_hr)
            label_vecs = torch.nn.functional.softmax(label_vecs, -1)
            ret_dict["label_vecs"] = label_vecs
            '''
            ##############
            # val: [T, B] <- [T, B, C]
            label_val, label_indices = torch.max(label_vecs, -1)
            label_indices = label_indices.view(label_indices.shape[0], label_indices.shape[1], 1)
            new_label_vecs = torch.zeros(label_vecs.shape).float().to(label_vecs.device)
            new_label_vecs.scatter_(2, label_indices, 1)
            # label_vecs[label_vecs > 0.5] = 1.
            noise = (torch.rand(label_vecs.shape) - 0.5) * 0.2
            label_vecs = new_label_vecs.to(label_vecs.device) + noise.to(label_vecs.device)
            ##############
            '''
            # [T, B, C] -> [B, T, C] -> [B, 1, T, C]
            label_vecs = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
            ###############################################

            images_sr = model_list[0](images_lr, label_vecs)

        elif self.args.arch in ABLATION_SET:

            cascade_images = images_lr

            images_sr = []

            if vis:
                aster_dict_hr = self.parse_crnn_data(
                    images_lr[:, :3, :, :] if not self.args.y_domain else images_lrraw[:, :3, :, :])
                # print("aster_dict_hr:", aster_dict_hr.shape)
                label_vecs_lr = aster[0]['model'](aster_dict_hr)
                label_vecs_lr = torch.nn.functional.softmax(label_vecs_lr, -1)

                aster_dict_hr = self.parse_crnn_data(
                    images_hr[:, :3, :, :] if not self.args.y_domain else images_hrraw[:, :3, :, :])
                label_vecs_hr = aster[0]['model'](aster_dict_hr)
                label_vecs_hr = torch.nn.functional.softmax(label_vecs_hr, -1)
                label_vecs_final_hr = label_vecs_hr.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

                ret_dict["label_vecs_hr"] = label_vecs_hr

            for m_iter in range(self.args.stu_iter):

                if self.args.tpg_share:
                    tpg_pick = 0
                else:
                    tpg_pick = m_iter

                stu_model = aster[1][tpg_pick]
                aster_dict_lr = self.parse_crnn_data(
                    cascade_images[:, :3, :, :] if not self.args.y_domain else images_lrraw[:, :3, :,
                                                                               :])  # cascade_images
                before = time.time()
                label_vecs_logits = stu_model(aster_dict_lr)
                after = time.time()

                ret_dict["duration"] += (after - before)

                label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)
                label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                ret_dict["label_vecs"] = label_vecs

                # image for cascading
                if self.args.for_cascading:
                    if i > 0:
                        cascade_images = nn.functional.interpolate(cascade_images,
                                                                   (
                                                                       self.config.TRAIN.height // self.scale_factor,
                                                                       self.config.TRAIN.width // self.scale_factor),
                                                                   mode='bicubic')
                        # cascade_images = model(cascade_images, label_vecs_final)
                        cascade_images[cascade_images > 1.0] = 1.0
                        cascade_images[cascade_images < 0.0] = 0.0

                        cascade_images = (cascade_images + images_lr) / 2

                if self.args.sr_share:
                    pick = 0
                else:
                    pick = m_iter

                if False:
                    # [N, C, H, W] -> [N, W]
                    probs, label = label_vecs_final.squeeze(2).max(1)

                    fg_probs, fg_label = label_vecs_final[:, 1:].squeeze(2).max(1)

                    # print("fg_label:", label, probs)

                    pseudo_logits = torch.zeros_like(label_vecs_final).to(label.device)

                    fg_cnt = 0
                    for batch_i in range(label.shape[0]):
                        for batch_w in range(label.shape[1]):
                            if label[batch_i, batch_w] > 0:
                                fg_cnt += 1

                            if label[batch_i, batch_w] == 28:
                                print("pseudo_logits", pseudo_logits[batch_i, 30, :, batch_w], label[batch_i, batch_w])
                                pseudo_logits[batch_i, 12, :, batch_w] = self.args.prob_insert
                                pseudo_logits[batch_i, 28, :, batch_w] = 1 - self.args.prob_insert
                            else:
                                pseudo_logits[batch_i, label[batch_i, batch_w], :, batch_w] = 1.

                    pseudo_probs, pseudo_label = pseudo_logits.squeeze(2).max(1)
                    print("pseudo_label:", pseudo_label, pseudo_probs)


                before = time.time()
                cascade_images, pr_weights = model_list[pick](images_lr if not self.args.for_cascading else cascade_images,
                                                              label_vecs_final.detach())  # label_vecs_final label_vecs_final label_vecs_gt.to(label_vecs_final.device)
                after = time.time()

                ret_dict["pr_weights"] = pr_weights

                # print("fps:", (after - before))
                ret_dict["duration"] += (after - before)

                if self.args.y_domain:
                    """
                    images_lrraw_bicubic = nn.functional.interpolate(images_lrraw,
                                                          (
                                                              self.config.TRAIN.height,
                                                              self.config.TRAIN.width),
                                                          mode='bicubic')

                    uv_domain = self.rgb_to_yuv(images_lrraw_bicubic[:, :3, ...])[:, 1:, ...]

                    cas_imgs = cascade_images[:, :1, ...]
                    cas_img_yuvs = torch.cat([cas_imgs, uv_domain], 1) #, images_lrraw_bicubic[:, 3:, ...]
                    cas_img_rgb = self.yuv_to_rgb(cas_img_yuvs)
                    cascade_images = torch.cat([cas_img_rgb, images_lrraw_bicubic[:, 3:]], 1)
                    """
                    ''''''
                    cas_img_yuvs = cascade_images[:, :3]
                    cas_img_rgb = self.yuv_to_rgb(cas_img_yuvs)
                    cascade_images = torch.cat([cas_img_rgb, cascade_images[:, 3:]], 1)

                    cas_img_yuvs = images_lr[:, :3]
                    cas_img_rgb = self.yuv_to_rgb(cas_img_yuvs)
                    images_lr = torch.cat([cas_img_rgb, images_lr[:, 3:]], 1)

                    cas_img_yuvs = images_hr[:, :3]
                    cas_img_rgb = self.yuv_to_rgb(cas_img_yuvs)
                    images_hr = torch.cat([cas_img_rgb, images_hr[:, 3:]], 1)

                    # images_lr = images_lrraw
                    # images_hr = images_hrraw

                images_sr.append(cascade_images)

                if vis_feature:
                    # [N, C, H, W] -> [N, C, W]
                    block_feature = model_list[-1].block["7"].mean(2)

            if vis:
                ret_dict["label_vecs_lr"] = label_vecs_lr
                ret_dict["label_vecs_final_hr"] = label_vecs_final_hr
                ret_dict["label_vecs_final"] = label_vecs_final
        else:
            if self.args.arch in ["srcnn", "rdn", "vdsr", 'han']:
                channel_num = 3
            else:
                channel_num = 4

            before = time.time()
            images_sr = model_list[0](images_lr[:, :channel_num, ...])
            after = time.time()

            ret_dict["duration"] += (after - before)

            if vis_feature:
                # [N, C, H, W] -> [N, C, W]
                block_feature = model_list[-1].block.mean(2)

        ret_dict["images_sr"] = images_sr
        if vis_feature:
            ret_dict["block_feature"] = block_feature

        # if vis:
        #     ret_dict["label_vecs_lr"] = label_vecs_lr
        #     ret_dict["label_vecs_final_hr"] = label_vecs_final_hr
        #     ret_dict["label_vecs_final"] = label_vecs_final

        return ret_dict

    def train(self):

        TP_Generator_dict = {
            "CRNN": self.CRNN_init,
            "OPT": self.TPG_init
        }

        tpg_opt = self.opt_TPG

        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init(0)
        model, image_crit = model_dict['model'], model_dict['crit']

        model_list = [model]
        if not self.args.sr_share:
            for i in range(self.args.stu_iter - 1):
                model_sep = self.generator_init(i+1)['model']
                model_list.append(model_sep)
        # else:
        #     model_list = [model]

        tensorboard_dir = os.path.join("tensorboard", self.vis_dir)
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        else:
            print("Directory exist, remove events...")
            os.popen("rm " + tensorboard_dir + "/*")

        self.results_recorder = SummaryWriter(tensorboard_dir)

        aster, aster_info = TP_Generator_dict[self.args.tpg](recognizer_path=None, opt=tpg_opt)

        test_bible = {}

        if self.args.test_model == "CRNN":
            crnn, aster_info = self.TPG_init(recognizer_path=None, opt=tpg_opt) if self.args.CHNSR else self.CRNN_init()
            crnn.eval()
            test_bible["CRNN"] = {
                        'model': crnn,
                        'data_in_fn': self.parse_OPT_data if self.args.CHNSR else self.parse_crnn_data,
                        'string_process': get_string_crnn
                    }

        elif self.args.test_model == "ASTER":
            aster_real, aster_real_info = self.Aster_init()
            aster_info = aster_real_info
            test_bible["ASTER"] = {
                'model': aster_real,
                'data_in_fn': self.parse_aster_data,
                'string_process': get_string_aster
            }

        elif self.args.test_model == "MORAN":
            moran = self.MORAN_init()
            if isinstance(moran, torch.nn.DataParallel):
                moran.device_ids = [0]
            test_bible["MORAN"] = {
                'model': moran,
                'data_in_fn': self.parse_moran_data,
                'string_process': get_string_crnn
            }

        # print("self.args.arch:", self.args.arch)
        aster_student = []
        if self.args.arch in ["tsrn_tl_wmask", "tsrn_tl"]:
            recognizer_path = os.path.join("/".join(self.resume.split("/")[:-1]), "recognizer_best.pth")
            if os.path.isfile(recognizer_path):
                aster_student, aster_stu_info = self.CRNN_init(recognizer_path=recognizer_path)
            else:
                aster_student, aster_stu_info = self.CRNN_init(recognizer_path=None)
            aster_student.train()
        elif self.args.arch in ABLATION_SET:
            aster_student = []
            stu_iter = self.args.stu_iter

            for i in range(stu_iter):
                recognizer_path = os.path.join("/".join(self.resume.split("/")[:-1]), "recognizer_best_acc_" + str(i) + ".pth")
                # print("recognizer_path:", recognizer_path)
                if os.path.isfile(recognizer_path):
                    aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=recognizer_path, opt=tpg_opt)
                else:
                    aster_student_, aster_stu_info = TP_Generator_dict[self.args.tpg](recognizer_path=None, opt=tpg_opt)

                if type(aster_student_) == list:
                    aster_student_ = aster_student_[i]

                aster_student_.train()
                aster_student.append(aster_student_)

        aster.eval()
        if self.args.use_label:
            aster.train()

        # Recognizer needs to be fixed:
        # aster
        if self.args.arch in ["tsrn_tl_wmask", "tsrn_tl"] + ABLATION_SET:
            if self.args.use_label:
                optimizer_G = self.optimizer_init(model_list + [aster], recognizer=aster_student)
            else:
                optimizer_G = self.optimizer_init(model_list, recognizer=aster_student)
        else:
            optimizer_G = self.optimizer_init(model_list)
        # for p in aster.parameters():
        #     p.requires_grad = False

        #print("cfg:", cfg.ckpt_dir)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []
        lr = cfg.lr

        for model in model_list:
            model.train()

        for epoch in range(cfg.epochs):

            for j, data in (enumerate(train_loader)):

                iters = len(train_loader) * epoch + j + 1

                if not self.args.go_test:
                    for model in model_list:
                        for p in model.parameters():
                            p.requires_grad = True
                    if self.args.syn:
                        images_hr, images_lr, images_hr, images_lr, label_strs, text_label, weighted_mask, weighted_tics = data
                    else:
                        if self.args.arch == "tsrn":
                            images_hr, images_lr, label_strs = data
                        elif self.args.arch == "sem_tsrn":
                            images_hr, images_lr, label_strs, word_vec = data
                        elif self.args.arch == "tsrn_c2f":
                            images_hr, images_lr, label_strs, image_coar_gt = data
                        elif self.args.arch == "tsrn_tl":
                            images_hr, images_lr, label_strs, label_vecs = data
                        elif self.args.arch == 'tsrn_tl_wmask':
                            images_hr, images_lr, label_strs, label_vecs, weighted_mask = data
                        elif self.args.arch in ABLATION_SET:
                            images_hrraw, images_pseudoLR, images_lrraw, images_HRy, images_lry, label_strs, label_vecs, weighted_mask, weighted_tics = data
                            text_label = label_vecs
                        else:
                            images_hr, images_lr, label_strs = data
                    # print("images_lr:", images_lr.shape)
                    ############# SynLR
                    # images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor, self.config.TRAIN.width // self.scale_factor), mode='bicubic')

                    if self.args.syn:
                        #images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                        #                                                 self.config.TRAIN.width // self.scale_factor),
                        #                                      mode='bicubic')
                        images_lr = images_lr.to(self.device)
                        images_hr = images_hr.to(self.device)
                    elif self.args.arch in ABLATION_SET:
                        if self.args.y_domain:

                            images_lrraw = images_lrraw.to(self.device)
                            images_hrraw = images_hrraw.to(self.device)

                            images_hr = images_HRy.to(self.device)
                            images_hr = torch.cat([images_hr[:, :3]] + [images_hrraw[:, 3:]], 1)

                            images_lr = images_lry.to(self.device)
                            images_lr = torch.cat([images_lr[:, :3]] + [images_lrraw[:, 3:]], 1)

                        else:
                            images_lr = images_lrraw.to(self.device)
                            images_hr = images_hrraw.to(self.device)
                            ############## SynLR ##############
                            # print("SynLR here...") remove afterwards
                            # images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor, self.config.TRAIN.width // self.scale_factor), mode='bicubic')

                    else:
                        images_lr = images_lr.to(self.device)
                        images_hr = images_hr.to(self.device)


                    if self.args.rotate_train:
                        # print("We are in rotate_train", self.args.rotate_train)
                        batch_size = images_lr.shape[0]

                        angle_batch = np.random.rand(batch_size) * self.args.rotate_train * 2 - self.args.rotate_train
                        arc = angle_batch / 180. * math.pi
                        rand_offs = torch.tensor(np.random.rand(batch_size)).float()

                        arc = torch.tensor(arc).float()

                        images_lr_origin = images_lr.clone()
                        images_hr_origin = images_hr.clone()

                        images_lr = self.torch_rotate_img(images_lr, arc, rand_offs)
                        images_hr = self.torch_rotate_img(images_hr, arc, rand_offs)

                        images_lr_ret = self.torch_rotate_img(images_lr.clone(), -arc, rand_offs)
                        images_hr_ret = self.torch_rotate_img(images_hr.clone(), -arc, rand_offs)
                        # print(images_lr.shape, images_hr.shape)

                    # print("iters:", iters)

                    loss_ssim = 0.
                    trans_quality_loss = torch.tensor(0.)
                    loss_tssim = torch.tensor(0.)
                    loss_color = torch.tensor(0.)

                    if self.args.arch == "tsrn":
                        image_sr = model(images_lr)
                        loss_img = loss_im = image_crit(image_sr, images_hr).mean() * 100
                        loss_recog_distill = torch.zeros(1)

                        if self.args.color_loss:

                            loss_color = torch.abs(images_lr.mean(-1).mean(-1) - image_sr.mean(-1).mean(-1)).mean() * 30
                            loss_img += loss_color

                            # print("loss_color:", loss_color)

                        if self.args.ssim_loss:
                            loss_ssim = (1 - distorted_ssim(image_sr, images_hr).mean()) * 10.
                            loss_img += loss_ssim

                        if self.args.tssim_loss:
                            cascade_images_sr_ret = model(images_lr_ret)

                            # image_sr.retain_grad()
                            cascade_images_ret = self.torch_rotate_img(cascade_images_sr_ret, arc, rand_offs)

                            loss_tssim = (1 - tri_ssim(image_sr,
                                                              cascade_images_ret, images_hr).mean()) * 10.
                            loss_img += loss_tssim

                            # loss_img += image_crit(cascade_images_sr_ret, images_hr_ret).mean() * 100

                        if self.args.mse_fuse:
                            cascade_images_sr_ret = model(images_lr_ret)

                            # image_sr.retain_grad()
                            cascade_images_ret = self.torch_rotate_img(cascade_images_sr_ret, -arc, rand_offs)

                            loss_tssim = image_crit(image_sr.clone(),
                                                              cascade_images_sr_ret.detach()).mean() * 100.
                            loss_img += loss_tssim

                    elif self.args.arch == "sem_tsrn":
                        # print("keys:", image_crit.keys())
                        image_sr, all_pred_vecs = model(images_lr, word_vec)

                        # print("shape:", image_sr.shape, image_masks.unsqueeze(1).shape)

                        loss_img = image_crit["image_loss"](image_sr, images_hr)
                        # print("loss:", loss_img.shape)
                        loss_img = loss_img.mean() * 100
                        loss_sem = image_crit["semantic_loss"]

                        loss_sem_cal = 0.
                        # for pred_vec in all_pred_vecs:
                        #    loss_sem_cal += loss_sem(pred_vec, word_vec) * 0.

                        # loss_sem_cal = torch.sum(loss_sem_cal)

                        loss_im = loss_img + loss_sem_cal
                        # print("loss_im:", loss_img.data.cpu().numpy(), "loss_sem:", loss_sem_cal.data.cpu().numpy())
                    elif self.args.arch == "tsrn_c2f":
                        # print("keys:", image_crit.keys())
                        image_sr, image_coar = model(images_lr)

                        loss_img = image_crit(image_sr, images_hr).mean() * 100
                        loss_coar = image_crit(image_coar, image_coar_gt).mean() * 100
                        # loss_sem = image_crit["semantic_loss"]
                        loss_im = loss_img + loss_coar
                    elif self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"]:

                        ###############################################
                        aster_dict_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                        label_vecs_logits = aster_student(aster_dict_lr)
                        label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)

                        aster_dict_hr = self.parse_crnn_data(images_hr[:, :3, :, :])
                        label_vecs_logits_hr = aster(aster_dict_hr).detach()
                        label_vecs_hr = torch.nn.functional.softmax(label_vecs_logits_hr, -1)

                        # label_vecs[label_vecs > 0.5] = 1.
                        # print("label_vecs:", np.unique(label_vecs.data.cpu().numpy()))
                        '''
                        ##############
                        # val: [T, B] <- [T, B, C]
                        label_val, label_indices = torch.max(label_vecs, -1)
                        label_indices = label_indices.view(label_indices.shape[0], label_indices.shape[1], 1)
                        new_label_vecs = torch.zeros(label_vecs.shape).float().to(label_vecs.device)
                        new_label_vecs.scatter_(2, label_indices, 1)
                        # label_vecs[label_vecs > 0.5] = 1.
                        noise = (torch.rand(label_vecs.shape) - 0.5) * 0.2
                        label_vecs = new_label_vecs.to(label_vecs.device) + noise.to(label_vecs.device)
                        ##############
                        '''

                        # print("text_label:", text_label)

                        # noise = (torch.rand(label_vecs.shape) - 0.5) * 0.2
                        # label_vecs += noise.to(label_vecs.device)
                        # [T, B, C] -> [B, T, C] -> [B, 1, T, C]
                        label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
                        ###############################################

                        image_sr = model(images_lr, label_vecs_final)

                        loss_img = image_crit(image_sr, images_hr, grad_mask=weighted_mask).mean() * 100
                        # loss_recog_distill# = torch.abs(label_vecs - label_vecs_hr).mean() * 100
                        loss_recog_distill = sem_loss(label_vecs, label_vecs_hr) * 100
                        loss_im = loss_img + loss_recog_distill

                    elif self.args.arch in ABLATION_SET:

                        aster_dict_hr = self.parse_crnn_data(images_hr[:, :3, :, :] if not self.args.y_domain else images_hrraw[:, :3, :, :])
                        label_vecs_logits_hr = aster(aster_dict_hr)
                        label_vecs_hr = torch.nn.functional.softmax(label_vecs_logits_hr, -1).detach()

                        # label_vecs_final_hr = label_vecs_hr.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

                        cascade_images = images_lr

                        loss_img = 0.
                        loss_recog_distill = 0.


                        for i in range(self.args.stu_iter):

                            cascade_images = cascade_images.detach()

                            if self.args.tpg_share:
                                tpg_pick = 0
                            else:
                                tpg_pick = i
                            stu_model = aster_student[tpg_pick]

                            aster_dict_lr = self.parse_crnn_data(cascade_images[:, :3, :, :] if not self.args.y_domain else images_lrraw[:, :3, :, :])

                            label_vecs_logits = stu_model(aster_dict_lr)
                            label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)

                            label_vecs_final = label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

                            # image for cascading
                            if self.args.sr_share:
                                pick = 0
                            else:
                                pick = i

                            # image for cascading
                            if self.args.for_cascading:
                                if i > 0:
                                    cascade_images = nn.functional.interpolate(cascade_images,
                                                                               (
                                                                                   self.config.TRAIN.height // self.scale_factor,
                                                                                   self.config.TRAIN.width // self.scale_factor),
                                                                               mode='bicubic')

                                    cascade_images[cascade_images > 1.0] = 1.0
                                    cascade_images[cascade_images < 0.0] = 0.0

                                    cascade_images = (cascade_images + images_lr) / 2

                                # print("cascade_images:", np.unique(cascade_images.data.cpu().numpy()), np.unique(images_hr.data.cpu().numpy()))

                            # else:
                            #     cascade_images = images_lr

                            # print("text_label:", text_label.shape, text_label.sum(2).shape)
                            if self.args.use_label:
                                # [B, L]
                                text_sum = text_label.sum(1).squeeze(1)
                                # print("text_sum:", text_sum.shape)
                                text_pos = (text_sum > 0).float().sum(1)
                                text_len = text_pos.reshape(-1)
                                predicted_length = torch.ones(label_vecs_logits.shape[1]) * label_vecs_logits.shape[0]

                                #fsup_sem_loss = ctc_loss(
                                #    label_vecs_logits.log_softmax(2),
                                #    weighted_mask.long().to(label_vecs_logits.device),
                                #    predicted_length.long().to(label_vecs_logits.device),
                                #    text_len.long()
                                #)

                                fsup_sem_loss = ctc_loss(
                                    label_vecs_logits_hr.log_softmax(2),
                                    weighted_mask.long().to(label_vecs_logits.device),
                                    predicted_length.long().to(label_vecs_logits.device),
                                    text_len.long()
                                )

                                loss_recog_distill_each = (fsup_sem_loss * Variable(weighted_tics.float()).to(fsup_sem_loss.device))# .mean()
                                # print('loss_recog_distill_each:', loss_recog_distill_each)
                                loss_recog_distill_each = loss_recog_distill_each.mean()
                                loss_recog_distill += loss_recog_distill_each

                            # images_pseudoLR = images_pseudoLR.to(self.device)

                            if self.args.results_rotate:
                                # print("We are in rotate_train", self.args.rotate_train)

                                batch_size = images_lr.shape[0]

                                angle_batch = np.random.rand(
                                    batch_size) * self.args.results_rotate_angle * 2 - self.args.results_rotate_angle
                                results_arc = angle_batch / 180. * math.pi
                                results_rand_offs = torch.tensor(np.random.rand(batch_size)).float()

                                cascade_images, ret_mid = model_list[pick](
                                                                                cascade_images,
                                                                                label_vecs_final.detach(),
                                                                                feature_arcs=results_arc,
                                                                                rand_offs=results_rand_offs
                                                                           )
                            else:
                                cascade_images, ret_mid = model_list[pick](images_lr if not self.args.for_cascading else cascade_images, label_vecs_final.detach())

                            # [N, C, H, W] -> [N, T, C]
                            # text_label = text_label.squeeze(2).permute(2, 0, 1)
                            if self.args.use_distill:
                                # print("label_vecs_hr:", label_vecs_hr.shape)
                                loss_recog_distill_each = sem_loss(label_vecs, label_vecs_hr) * 100 #100
                                loss_recog_distill += loss_recog_distill_each  # * (1 + 0.5 * i)

                                # print("pr_weights:", pr_weights.shape, label_masks.shape, label_vecs_final[:, 1:, ...].max(1)[0].shape)

                                # feature_recov_each = torch.abs(pr_weights_mask - label_masks).mean() * 100
                                # print("feature_recov_each:", feature_recov_each)

                                # loss_recog_distill += feature_recov_each

                            im_quality_loss = image_crit(cascade_images, images_hr)

                            if self.args.training_stablize:
                                im_quality_loss = self.loss_stablizing(im_quality_loss)

                            loss_img_each = im_quality_loss.mean() * 100

                            if self.args.learning_STN:
                                in_feat = ret_mid["in_feat"]
                                trans_feat = ret_mid["trans_feat"]

                                trans_quality_loss = image_crit(in_feat, trans_feat).mean()
                                loss_img_each += trans_quality_loss

                            loss_img += loss_img_each * (1 + i * 0.5)

                            if self.args.ssim_loss:

                                loss_ssim = (1 - ssim(cascade_images, images_hr).mean()) * 10.
                                loss_img += loss_ssim

                            if self.args.tssim_loss:
                                cascade_images_sr_ret, srret_mid = model_list[pick](images_lr_ret, label_vecs_final.detach())
                                cascade_images_ret = self.torch_rotate_img(cascade_images_sr_ret, arc, rand_offs)
                                loss_tssim = (1 - tri_ssim(cascade_images_ret, cascade_images, images_hr).mean()) * 10.
                                loss_img += loss_tssim

                            if self.args.y_domain:

                                cas_img_yuvs = cascade_images[:, :3]
                                cas_img_rgb = self.yuv_to_rgb(cas_img_yuvs)
                                cascade_images = torch.cat([cas_img_rgb, cascade_images[:, 3:]], 1)
                                pass
                                images_lr = images_lrraw
                                images_hr = images_hrraw

                            if _DEBUG and iters % 20 == 0 and self.args.arch == "tsrn_tl_cascade":

                                pr_weights = ret_mid["pr_weights"]
                                pr_weights_gt = ret_mid["pr_weights_gt"]
                                spatial_t_emb = ret_mid["spatial_t_emb"]
                                spatial_t_emb_gt = ret_mid["spatial_t_emb_gt"]

                                attw = pr_weights[0]
                                # [L, C]
                                t_label = text_label[0].permute(1, 2, 0)[0][:, 1:].sum(1)
                                t_pred = label_vecs_final[0].permute(1, 2, 0)[0][:, 1:].sum(1)

                                lr_image = images_lr[0].permute(1, 2, 0)

                                # print("t_label:", t_label.shape, t_label)
                                sr_image = nn.functional.interpolate(cascade_images,
                                                                           (
                                                                               self.config.TRAIN.height // self.scale_factor,
                                                                               self.config.TRAIN.width // self.scale_factor),
                                                                           mode='bicubic')[0].permute(1, 2, 0)
                                # hr_image = images_hr[0].permute(1, 2, 0)

                                images_hr_resize = nn.functional.interpolate(images_hr,
                                                                           (
                                                                               self.config.TRAIN.height // self.scale_factor,
                                                                               self.config.TRAIN.width // self.scale_factor),
                                                                           mode='bicubic')

                                hr_image = images_hr_resize[0].permute(1, 2, 0)

                                # print("attw:", attw.shape)
                                w, h = 4, 8
                                W, H = 64, 16
                                vis_im = np.ones((h * (H + 1), w * (W + 1), 3)) * 255

                                cnt = 0

                                fg_mask = np.zeros((H, W))

                                for i in range(h):
                                    for j_w in range(w):
                                        if cnt < attw.shape[-1]:
                                            # print("attw:", attw[..., cnt].shape)
                                            attw_ = attw[..., cnt].reshape(H, W).data.cpu().numpy()
                                            # Normalize the weights
                                            # print("pred:", t_pred.shape)
                                            if t_pred[cnt] > 0.5:
                                                attw_ = attw_ / (np.max(attw_) + 1e-10)
                                                attw_ = attw_ * (attw_ > 0.4)

                                            vis_im[i * (H + 1):(i + 1) * (H + 1) - 1,
                                            j_w * (W + 1):(j_w + 1) * (W + 1) - 1] = (attw_[..., None] * 255).astype(np.int8)

                                            if t_label[cnt] > 0:
                                                fg_mask += attw_

                                            cnt += 1
                                            # print("cnt:", cnt, i * H, (i+1) * H, j * W, (j+1) * W)

                                        elif cnt == attw.shape[-1] + 2:

                                            insert = lr_image[..., :3].data.cpu().numpy() * 255
                                            insert[insert > 255] = 255

                                            vis_im[i * (H + 1):(i + 1) * (H + 1) - 1,
                                            j_w * (W + 1):(j_w + 1) * (W + 1) - 1] = (insert).astype(np.int8)

                                            cnt += 1

                                        elif cnt == attw.shape[-1] + 3:

                                            insert = insert = lr_image[..., :3].data.cpu().numpy() * 255
                                            insert[insert > 255] = 255

                                            vis_im[i * (H + 1):(i + 1) * (H + 1) - 1,
                                            j_w * (W + 1):(j_w + 1) * (W + 1) - 1] = (insert).astype(np.int8)

                                            cnt += 1

                                        elif cnt == attw.shape[-1] + 4:

                                            # print("hr_image:", np.unique(hr_image[..., :3].data.cpu().numpy()))

                                            insert = hr_image[..., :3].data.cpu().numpy() * 255
                                            insert[insert > 255] = 255

                                            vis_im[i * (H + 1):(i + 1) * (H + 1) - 1,
                                            j_w * (W + 1):(j_w + 1) * (W + 1) - 1] = (insert).astype(np.int8)

                                            cnt += 1

                                        else:
                                            cnt += 1

                                cv2.imwrite("saving.jpg", cv2.cvtColor(vis_im.astype(np.uint8), cv2.COLOR_RGB2BGR))
                                cv2.imwrite("saving_mask.jpg", cv2.cvtColor((fg_mask / (t_label.sum(0).data.cpu().numpy() + 0.001) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


                            if iters % 5 == 0 and i == self.args.stu_iter - 1:

                                self.results_recorder.add_scalar('loss/distill', float(loss_recog_distill_each.data) * 100,
                                                                 global_step=iters)

                                self.results_recorder.add_scalar('loss/SR', float(loss_img_each.data) * 100,
                                                                 global_step=iters)

                                self.results_recorder.add_scalar('loss/SSIM', float(loss_ssim) * 100,
                                                                 global_step=iters)

                        loss_im = loss_img + loss_recog_distill


                    else:
                        if self.args.arch in ["srcnn", "rdn", "vdsr"]:
                            channel_num = 3
                        else:
                            channel_num = 4

                        image_sr = model(images_lr[:, :channel_num, ...])

                        if self.args.arch == "tbsrn" and not self.args.CHNSR:
                            loss_img = loss_im = image_crit(image_sr, images_hr, label_strs).mean() * 100

                        else:
                            loss_img = loss_im = image_crit(image_sr, images_hr[:, :channel_num, ...]).mean() * 100

                        if self.args.ssim_loss:
                            loss_ssim = (1 - ssim(image_sr, images_hr).mean()) * 10.
                            loss_img += loss_ssim

                        if self.args.tssim_loss:
                            cascade_images_sr_ret = model(images_lr_ret[:, :channel_num, ...])
                            cascade_images_ret = self.torch_rotate_img(cascade_images_sr_ret, arc, rand_offs)

                            loss_tssim = (1 - tri_ssim(image_sr,
                                                              cascade_images_sr_ret, images_hr).mean()) * 10.
                            loss_img += loss_tssim

                        loss_recog_distill = torch.zeros(1)
                    # loss_img.requ ires_grad=True
                    # grad_ = torch.autograd.grad(loss_im, label_vecs_logits)

                    #label_vecs_logits.retain_grad()
                    #label_vecs_logits_hr.retain_grad()
                    #loss_recog_distill.retain_grad()
                    #loss_img.retain_grad()

                    optimizer_G.zero_grad()
                    loss_im.backward()

                    # print("image_sr:", image_sr.grad)

                    #print("label_vecs_logits_grad:",
                    #      label_vecs_logits.grad.shape,
                    #      # label_vecs_logits_hr.grad,
                    #      loss_recog_distill.grad.shape,
                    #      loss_img.grad.shape
                    #      )
                    for model in model_list:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optimizer_G.step()
                    if iters % 5 == 0:

                        self.results_recorder.add_scalar('loss/total', float(loss_im.data) * 100,
                                                    global_step=iters)


                    # torch.cuda.empty_cache()
                    if iters % cfg.displayInterval == 0:
                        print('[{}]\t'
                              'Epoch: [{}][{}/{}]\t'
                              'vis_dir={:s}\t'
                              'loss_total: {:.3f} \t'
                              'loss_im: {:.3f} \t'
                              'loss_teaching: {:.3f} \t'
                              'loss_tssim: {:.3f} \t'
                              '{:.3f} \t'
                              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                      epoch, j + 1, len(train_loader),
                                      self.vis_dir,
                                      float(loss_im.data),
                                      float(loss_img.data),
                                      float(loss_recog_distill.data),
                                      float(loss_tssim.data),
                                      lr))

                if iters % cfg.VAL.valInterval == 0 or self.args.go_test:
                    print('======================================================')
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        print('evaling %s' % data_name)
                        for model in model_list:
                            model.eval()
                            for p in model.parameters():
                                p.requires_grad = False

                        if self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"]:
                            aster_student.eval()
                            for p in aster_student.parameters():
                                p.requires_grad = False
                        elif self.args.arch in ABLATION_SET:
                            for stu in aster_student:
                                stu.eval()
                                for p in stu.parameters():
                                    p.requires_grad = False

                        else:
                            aster_student = aster

                        # Tuned TPG for recognition:
                        # test_bible[self.args.test_model]['model'] = aster_student[-1]

                        metrics_dict = self.eval(
                            model_list,
                            val_loader,
                            image_crit,
                            iters,
                            [test_bible[self.args.test_model], aster_student, aster], # aster_student[0]test_bible[self.args.test_model]
                            aster_info,
                            data_name
                        )

                        for key in metrics_dict:
                            # print(metrics_dict)
                            if key in ["psnr_avg", "ssim_avg", "accuracy"]:
                                self.results_recorder.add_scalar('eval/' + key + "_" + data_name, float(metrics_dict[key]),
                                                    global_step=iters)

                        if self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"]:
                            for p in aster_student.parameters():
                                p.requires_grad = True
                            aster_student.train()
                        elif self.args.arch in ABLATION_SET:
                            for stu in aster_student:
                                for p in stu.parameters():
                                    p.requires_grad = True
                                stu.train()

                        for model in model_list:
                            for p in model.parameters():
                                p.requires_grad = True
                            model.train()

                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:
                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))

                        # if self.args.go_test:
                        #     break
                    if self.args.go_test:
                        break
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model')
                        self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, True, converge_list, recognizer=aster_student)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model_list, epoch, iters, best_history_acc, best_model_info, False, converge_list, recognizer=aster_student)
            if self.args.go_test:
                break

    def eval(self, model_list, val_loader, image_crit, index, aster, aster_info, data_name=None):

        n_correct = 0
        n_correct_lr = 0
        n_correct_hr = 0
        sum_images = 0
        metric_dict = {
                       'psnr_lr': [],
                       'ssim_lr': [],
                       'cnt_psnr_lr': [],
                       'cnt_ssim_lr': [],
                       'psnr': [],
                       'ssim': [],
                       'cnt_psnr': [],
                       'cnt_ssim': [],
                       'accuracy': 0.0,
                       'psnr_avg': 0.0,
                       'ssim_avg': 0.0,
                        'edis_LR': [],
                        'edis_SR': [],
                        'edis_HR': [],
                        'LPIPS_VGG_LR': [],
                        'LPIPS_VGG_SR': []
                       }

        counters = {i: 0 for i in range(self.args.stu_iter)}
        wrong_cnt = 0

        if vis:
            vis_dir = os.path.join("image_" + self.resume.split("/")[-2], data_name)# + "B_" + str(self.args.prob_insert)
            if not os.path.isdir(vis_dir):
                os.makedirs(vis_dir)
            identity_file = os.path.join(vis_dir, "identity.txt")

            i_f = open(identity_file, "a+")

            rec_file = os.path.join(vis_dir, data_name + ".txt")
            rec_f = open(rec_file, "w")

        if vis_feature:
            vis_dir = os.path.join("image_" + self.resume.split("/")[-2], data_name)
            if not os.path.isdir(vis_dir):
                os.makedirs(vis_dir)
            pkl_file = os.path.join("image_" + self.resume.split("/")[-2] + '_' + data_name, "feature.pkl")
            pkl_f = open(pkl_file, "wb")

            pkl_database = {}

            for i in range(37):
                pkl_database[str(i)] = []

        image_counter = 0
        rec_str = ""

        sr_infer_time = 0

        #############################################
        # Print the computational cost and param size
        # self.cal_all_models(model_list, aster[1])
        #############################################

        for i, data in (enumerate(val_loader)):

            if self.args.syn:
                # images_hr, images_lr, label_strs = data
                images_hrraw, images_lrraw, images_HRy, images_lry, \
                label_strs, label_vecs_gt, weighted_mask, weighted_tics = data
            else:
                if self.args.arch == "tsrn":
                    images_hr, images_lr, label_strs = data
                elif self.args.arch == "sem_tsrn":
                    images_hr, images_lr, label_strs, word_vec = data
                elif self.args.arch == "tsrn_c2f":
                    images_hr, images_lr, label_strs, image_mx = data
                elif self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"] + ABLATION_SET:
                    images_hrraw, images_lrraw, images_HRy, images_lry, label_strs, label_vecs_gt = data
                else:
                    images_hr, images_lr, label_strs = data


            # print("label_strs:", label_strs)

            if self.args.y_domain:

                images_lrraw = images_lrraw.to(self.device)
                images_hrraw = images_hrraw.to(self.device)

                images_hr = images_HRy.to(self.device)
                images_hr = torch.cat([images_hr[:, :3]] + [images_hrraw[:, 3:]], 1)

                images_lr = images_lry.to(self.device)
                images_lr = torch.cat([images_lr[:, :3]] + [images_lrraw[:, 3:]], 1)

            elif self.args.arch in ABLATION_SET:

                if self.args.y_domain:

                    images_lrraw = images_lrraw.to(self.device)
                    images_hrraw = images_hrraw.to(self.device)

                    images_hr = images_HRy.to(self.device)
                    images_hr = torch.cat([images_hr[:, :3]] + [images_hrraw[:, 3:]], 1)

                    images_lr = images_lry.to(self.device)
                    images_lr = torch.cat([images_lr[:, :3]] + [images_lrraw[:, 3:]], 1)

                else:
                    images_lr = images_lrraw.to(self.device)
                    images_hr = images_hrraw.to(self.device)

            else:
                if self.args.syn:
                    images_lr = images_lrraw.to(self.device)
                    images_hr = images_hrraw.to(self.device)
                else:
                    images_lr = images_lr.to(self.device)
                    images_hr = images_hr.to(self.device)

            val_batch_size = images_lr.shape[0]
            # images_hr = images_hr.to(self.device)

            ret_dict = self.model_inference(images_lr, images_hr, model_list, aster, i)

            # time_after = time.time()
            # print(ret_dict["duration"])
            sr_infer_time += ret_dict["duration"]

            if vis_feature:
                block_feature = ret_dict["block_feature"]

            if vis:

                label_vecs_lr = ret_dict["label_vecs_lr"] if "label_vecs_lr" in ret_dict.keys() else None
                label_vecs_final_hr = ret_dict["label_vecs_final_hr"] if 'label_vecs_final_hr' in ret_dict.keys() else None
                label_vecs_final = ret_dict["label_vecs_final"] if 'label_vecs_final' in ret_dict.keys() else None
                label_vecs_hr = ret_dict["label_vecs_hr"] if 'label_vecs_hr' in ret_dict.keys() else None

                label_vecs = ret_dict["label_vecs"]
            images_sr = ret_dict["images_sr"]

            if _DEBUG:
                pr_weights = ret_dict["pr_weights"]

            # print("images_lr:", images_lr.device, images_hr.device)

            aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, :, :])
            aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, :, :])

            if self.args.test_model == "MORAN":
                # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                # LR
                aster_output_lr = aster[0]["model"](
                    aster_dict_lr[0],
                    aster_dict_lr[1],
                    aster_dict_lr[2],
                    aster_dict_lr[3],
                    test=True,
                    debug=True
                )
                # HR
                aster_output_hr = aster[0]["model"](
                    aster_dict_hr[0],
                    aster_dict_hr[1],
                    aster_dict_hr[2],
                    aster_dict_hr[3],
                    test=True,
                    debug=True
                )
            else:
                aster_output_lr = aster[0]["model"](aster_dict_lr)
                aster_output_hr = aster[0]["model"](aster_dict_hr)

            if type(images_sr) == list:
                predict_result_sr = []
                for i in range(self.args.stu_iter):
                    image = images_sr[i]
                    aster_dict_sr = aster[0]["data_in_fn"](image[:, :3, :, :])
                    if self.args.test_model == "MORAN":
                        # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                        aster_output_sr = aster[0]["model"](
                            aster_dict_sr[0],
                            aster_dict_sr[1],
                            aster_dict_sr[2],
                            aster_dict_sr[3],
                            test=True,
                            debug=True
                        )
                    else:
                        aster_output_sr = aster[0]["model"](aster_dict_sr)
                    # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                    if self.args.test_model == "CRNN":
                        predict_result_sr_ = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
                    elif self.args.test_model == "ASTER":
                        predict_result_sr_, _ = aster[0]["string_process"](
                            aster_output_sr['output']['pred_rec'],
                            aster_dict_sr['rec_targets'],
                            dataset=aster_info
                        )
                    elif self.args.test_model == "MORAN":
                        preds, preds_reverse = aster_output_sr[0]
                        _, preds = preds.max(1)
                        sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                        predict_result_sr_ = [pred.split('$')[0] for pred in sim_preds]

                    predict_result_sr.append(predict_result_sr_)

                img_lr = torch.nn.functional.interpolate(images_lr, images_hr.shape[-2:], mode="bicubic")
                img_sr = torch.nn.functional.interpolate(images_sr[-1], images_hr.shape[-2:], mode="bicubic")

                metric_dict['psnr'].append(self.cal_psnr(img_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(img_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(img_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            else:

                aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])
                if self.args.test_model == "MORAN":
                    # aster_output_sr = aster[0]["model"](*aster_dict_sr)
                    aster_output_sr = aster[0]["model"](
                        aster_dict_sr[0],
                        aster_dict_sr[1],
                        aster_dict_sr[2],
                        aster_dict_sr[3],
                        test=True,
                        debug=True
                    )
                else:
                    aster_output_sr = aster[0]["model"](aster_dict_sr)
                # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()
                if self.args.test_model == "CRNN":
                    predict_result_sr = aster[0]["string_process"](aster_output_sr, self.args.CHNSR)
                elif self.args.test_model == "ASTER":
                    predict_result_sr, _ = aster[0]["string_process"](
                        aster_output_sr['output']['pred_rec'],
                        aster_dict_sr['rec_targets'],
                        dataset=aster_info
                    )
                elif self.args.test_model == "MORAN":
                    preds, preds_reverse = aster_output_sr[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
                    predict_result_sr = [pred.split('$')[0] for pred in sim_preds]

                img_lr = torch.nn.functional.interpolate(images_lr, images_sr.shape[-2:], mode="bicubic")

                metric_dict['psnr'].append(self.cal_psnr(images_sr[:, :3], images_hr[:, :3]))
                metric_dict['ssim'].append(self.cal_ssim(images_sr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_SR"].append(lpips_vgg(images_sr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

                metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
                metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

                metric_dict["LPIPS_VGG_LR"].append(lpips_vgg(img_lr[:, :3].cpu(), images_hr[:, :3].cpu()).data.numpy()[0].reshape(-1)[0])

            if self.args.test_model == "CRNN":
                predict_result_lr = aster[0]["string_process"](aster_output_lr, self.args.CHNSR)
                predict_result_hr = aster[0]["string_process"](aster_output_hr, self.args.CHNSR)
            elif self.args.test_model == "ASTER":
                predict_result_lr, _ = aster[0]["string_process"](
                    aster_output_lr['output']['pred_rec'],
                    aster_dict_lr['rec_targets'],
                    dataset=aster_info
                )
                predict_result_hr, _ = aster[0]["string_process"](
                    aster_output_hr['output']['pred_rec'],
                    aster_dict_hr['rec_targets'],
                    dataset=aster_info
                )
            elif self.args.test_model == "MORAN":
                ### LR ###
                preds, preds_reverse = aster_output_lr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
                predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

                ### HR ###
                preds, preds_reverse = aster_output_hr[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
                predict_result_hr = [pred.split('$')[0] for pred in sim_preds]

            cnt = 0


            filter_mode = 'chinese' if self.args.CHNSR else 'lower'

            if vis_feature:
                # [w_t, b, c] -> [b, c, 1, w_t]
                aster_output_sr = aster_output_sr.permute(1, 2, 0)[:, :, None, :]
                b, c, w = block_feature.shape
                b, c_t, _, w_t = aster_output_sr.shape

                block_feature = block_feature[:, :, None, :]

                block_feature = torch.nn.functional.interpolate(block_feature, (1, w_t))
                block_feature = block_feature.squeeze(2)

                aster_output_sr = torch.nn.functional.softmax(aster_output_sr, 1)

                max_value, max_indice = aster_output_sr.max(1)
                max_value = max_value.view(-1).data.cpu().numpy()
                max_indice = max_indice.view(-1).data.cpu().numpy()

                block_feature = block_feature.permute(0, 2, 1).reshape(b * w_t, c).data.cpu().numpy()
                for i in range(max_value.shape[0]):
                    if max_value[i] > 0.8 and max_indice[i] > 0:
                        pkl_database[str(max_indice[i])].append({"score": max_value[i], "feature": block_feature[i]})

            for batch_i in range(images_lr.shape[0]):

                label = label_strs[batch_i]

                image_counter += 1
                rec_str += str(image_counter) + ".jpg," + label + "\n"

                if self.args.arch in ABLATION_SET:
                    for k in range(self.args.stu_iter):
                        if str_filt(predict_result_sr[k][batch_i], filter_mode) == str_filt(label, filter_mode):
                            counters[k] += 1
                        if self.args.CHNSR:
                            pred_str, text_label = str_filt(predict_result_sr[k][batch_i], filter_mode), str_filt(label, filter_mode)
                            edis_SR = editdistance.eval(pred_str, text_label) / float(max(len(pred_str), len(text_label)) + 1e-10)
                            metric_dict["edis_SR"].append(edis_SR)
                else:

                    if self.args.CHNSR:
                        pred_str, text_label = str_filt(predict_result_sr[batch_i], filter_mode), str_filt(label, filter_mode)
                        edis_SR = editdistance.eval(pred_str, text_label) / float(max(len(pred_str), len(text_label)) + 1e-10)
                        metric_dict["edis_SR"].append(edis_SR)
                    if str_filt(predict_result_sr[batch_i], filter_mode) == str_filt(label, filter_mode):
                        n_correct += 1
                    else:
                        iswrong = True

                if self.args.CHNSR:
                    pred_str, text_label = str_filt(predict_result_lr[batch_i], filter_mode), str_filt(label, filter_mode)
                    edis_LR = editdistance.eval(pred_str, text_label) / float(max(len(pred_str), len(text_label)) + 1e-10)
                    metric_dict["edis_LR"].append(edis_LR)
                if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_lr += 1
                else:
                    iswrong = True

                if self.args.CHNSR:
                    pred_str, text_label = str_filt(predict_result_hr[batch_i], filter_mode), str_filt(label, filter_mode)
                    edis_HR = editdistance.eval(pred_str, text_label) / float(max(len(pred_str), len(text_label)) + 1e-10)
                    metric_dict["edis_HR"].append(edis_HR)
                if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                    n_correct_hr += 1
                else:
                    iswrong = True

                if vis:

                    if self.args.arch in ABLATION_SET:
                        all_label_vecs = {
                            'lr': label_vecs_lr.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2),
                            'sr': label_vecs.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2),
                            'hr': label_vecs_hr.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2),
                            # 'tl': label_vecs_gt,
                        }

                    sr, lr, hr = images_sr[-1][batch_i, :3, :, :] if type(images_sr) is list else images_sr[batch_i, :3, :, :], img_lr[batch_i, :3, :, :], images_hr[batch_i, :3, :, :]

                    sr = np.transpose(sr.data.cpu().numpy() * 255, (1, 2, 0))
                    lr = np.transpose(lr.data.cpu().numpy() * 255, (1, 2, 0))
                    hr = np.transpose(hr.data.cpu().numpy() * 255, (1, 2, 0))

                    sr[sr > 255] = 255
                    sr[sr < 0] = 0

                    lr[lr > 255] = 255
                    lr[lr < 0] = 0

                    hr[hr > 255] = 255
                    hr[hr < 0] = 0

                    sr = sr.astype(np.uint8)
                    lr = lr.astype(np.uint8)
                    hr = hr.astype(np.uint8)

                    lr = cv2.resize(lr, (self.config.TRAIN.width, self.config.TRAIN.height), interpolation=cv2.INTER_CUBIC)

                    ####################################################################
                    if not os.path.isdir(vis_dir + "_LR"):
                        os.makedirs(vis_dir + "_LR")
                    if not os.path.isdir(vis_dir + "_SR"):
                        os.makedirs(vis_dir + "_SR")

                    LR_image = os.path.join(vis_dir + "_LR", str(image_counter) + '.jpg')
                    SR_image = os.path.join(vis_dir + "_SR", str(image_counter) + '_' + label_strs[batch_i].split(".")[0] + '.jpg')

                    cv2.imwrite(LR_image, cv2.cvtColor(lr.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(SR_image, cv2.cvtColor(sr.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    if batch_i == 0:
                        print(SR_image)
                    ####################################################################

                    sr = cv2.resize(sr, (self.config.TRAIN.width, self.config.TRAIN.height), interpolation=cv2.INTER_CUBIC)
                    hr = cv2.resize(hr, (self.config.TRAIN.width, self.config.TRAIN.height), interpolation=cv2.INTER_CUBIC)

                    paddimg_im = np.zeros((sr.shape[0] + lr.shape[0] + hr.shape[0] + 20, sr.shape[1], 3))
                    paddimg_im[:lr.shape[0], :lr.shape[1], :] = lr
                    paddimg_im[lr.shape[0] + 5:lr.shape[0] + sr.shape[0] + 5, :sr.shape[1], :] = sr
                    paddimg_im[lr.shape[0] + sr.shape[0] + 10:lr.shape[0] + sr.shape[0] + 10 + hr.shape[0], :hr.shape[1], :] = hr
                    # if iswrong:

                    file_name = "lr_sr_hr_" + \
                                str(image_counter) + \
                                '_' + predict_result_lr[batch_i] + '_' + \
                                (predict_result_sr[-1][batch_i] if type(images_sr) is list else predict_result_sr[batch_i]) + '_' + \
                                predict_result_hr[batch_i] + "_" + label.lower() + '_' + label_strs[batch_i] + '.jpg'
                    cv2.imwrite(os.path.join(vis_dir, file_name), cv2.cvtColor(paddimg_im.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    wrong_cnt += 1

            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / (len(metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(metric_dict['ssim']) / (len(metric_dict['psnr']) + 1e-10)

        psnr_avg_lr = sum(metric_dict['psnr_lr']) / (len(metric_dict['psnr_lr']) + 1e-10)
        ssim_avg_lr = sum(metric_dict['ssim_lr']) / (len(metric_dict['ssim_lr']) + 1e-10)

        edis_LR = sum(metric_dict['edis_LR']) / (len(metric_dict['edis_LR']) + 1e-10)
        edis_SR = sum(metric_dict['edis_SR']) / (len(metric_dict['edis_SR']) + 1e-10)
        edis_HR = sum(metric_dict['edis_HR']) / (len(metric_dict['edis_HR']) + 1e-10)

        lpips_vgg_lr = sum(metric_dict["LPIPS_VGG_LR"]) / (len(metric_dict['LPIPS_VGG_LR']) + 1e-10)
        lpips_vgg_sr = sum(metric_dict["LPIPS_VGG_SR"]) / (len(metric_dict['LPIPS_VGG_SR']) + 1e-10)

        print('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              'LPIPS {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      0, 0,
                      float(psnr_avg), float(ssim_avg), float(lpips_vgg_sr)))

        print('[{}]\t'
              'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
              'LPIPS_LR {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg_lr), float(ssim_avg_lr), float(lpips_vgg_lr)))

        print('save display images')
        # self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)

        if self.args.arch in ABLATION_SET:
            acc = {i: 0 for i in range(self.args.stu_iter)}
            for i in range(self.args.stu_iter):
                acc[i] = round(counters[i] / sum_images, 4)
        else:
            accuracy = round(n_correct / sum_images, 4)
        accuracy_lr = round(n_correct_lr / sum_images, 4)
        accuracy_hr = round(n_correct_hr / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)

        if self.args.arch in ABLATION_SET:
            for i in range(self.args.stu_iter):
                print('sr_accuray_iter' + str(i) + ': %.2f%%' % (acc[i] * 100))
            accuracy = acc[self.args.stu_iter-1]

        else:
            print('sr_accuray: %.2f%%' % (accuracy * 100))

        # print('sr_NED: %.4f' % (edis_SR))
        print('lr_accuray: %.2f%%' % (accuracy_lr * 100))
        # print('lr_NED: %.4f' % (edis_LR))
        print('hr_accuray: %.2f%%' % (accuracy_hr * 100))
        # print('hr_NED: %.4f' % (edis_HR))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        # if self.args.arch in ["tsrn_tl", "tsrn_tl_wmask"]:
        #     aster[1].train()

        inference_time = sum_images / sr_infer_time
        print("AVG inference:", inference_time)
        print("sum_images:", sum_images)

        if vis:
            i_f.close()
            rec_f.write(rec_str)
            rec_f.close()

        if vis_feature:
            pickle.dump(pkl_database, pkl_f)
            pkl_f.close()
        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        test_data, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        # print(sum(p.numel() for p in moran.parameters()))
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        sr_time = 0
        for i, data in (enumerate(test_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_hr)

            # images_sr = images_lr
            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr["images"])
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input["images"])
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(pred_str_sr, label_strs):
                if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                    n_correct += 1
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{}/{}]\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader), ))
            # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((256, 32), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                     debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            print(pred_str_lr, '===>', pred_str_sr)
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)


if __name__ == '__main__':
    embed()
