import torch
import sys
import time
import os

import lmdb, io

from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
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

from ptflops import get_model_complexity_info
import string

vis = True

TEST_MODEL = "MORAN"
sem_loss = SemanticLoss()
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')

ssim = ssim_psnr.SSIM()

_DEBUG = True

class TextDegrade(base.TextBase):

    def train(self):

        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init(0)
        model, image_crit = model_dict['model'], model_dict['crit']

        model_deblur_dict = self.generator_init(0)
        model_deblur, image_deblur_crit = model_deblur_dict['model'], model_deblur_dict['crit']

        ckpt_degrade_name = os.path.join(self.resume, "model_degrade.pth")
        ckpt_deblur_name = os.path.join(self.resume, "model_deblur.pth")

        if os.path.isdir(self.resume):
            print("Loading resume ckpt...", self.resume)
            if os.path.isfile(ckpt_degrade_name):
                print("Loading resume ckpt...", ckpt_degrade_name)
                model.load_state_dict(torch.load(ckpt_degrade_name))
            if os.path.isfile(ckpt_degrade_name):
                print("Loading resume ckpt...", ckpt_deblur_name)
                model_deblur.load_state_dict(torch.load(ckpt_deblur_name))
        model.train()
        model_deblur.train()

        tensorboard_dir = os.path.join("tensorboard", self.vis_dir)
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        else:
            print("Directory exist, remove events...")
            os.popen("rm " + tensorboard_dir + "/*")

        self.results_recorder = SummaryWriter(tensorboard_dir)

        optimizer_G = self.optimizer_init([model, model_deblur])
        lr = cfg.lr

        if self.args.go_test:
            model_deblur.eval()
            model.eval()
            tar_dir = "./CSVTR_DR_test"
            self.test(model, val_loader_list[0], tar_dir)
            return

        for epoch in range(cfg.epochs):

            for j, data in (enumerate(train_loader)):

                iters = len(train_loader) * epoch + j + 1

                images_hr, images_lr, label_strs = data

                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                images_dr = model(images_hr)
                images_sr = model_deblur(images_dr)

                loss_im = image_crit(images_dr, images_lr).mean() * 100
                loss_im_delur = image_crit(images_sr, images_hr).mean() * 100

                loss_total = loss_im + loss_im_delur

                optimizer_G.zero_grad()
                loss_total.backward()

                # for model in model_list:
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
                          'loss_im: {:.3f} \t'
                          'loss_im_delur: {:.3f} \t'
                          '{:.3f} \t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_dir,
                                  float(loss_im.data),
                                  float(loss_im_delur.data),
                                  lr))

                if iters % cfg.saveInterval == 0:
                    ckpt_path = os.path.join('ckpt', self.vis_dir)
                    if not os.path.exists(ckpt_path):
                        os.mkdir(ckpt_path)
                    print("Saving checkpoint", os.path.join(ckpt_path))
                    torch.save(model.state_dict(), ckpt_degrade_name)
                    torch.save(model_deblur.state_dict(), ckpt_deblur_name)

    def image2bytes(self, image_pil):
        imgByteArr = io.BytesIO()
        image_pil.save(imgByteArr, format="PNG") #
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr

    def write2lmdb(self, lmdb_txn, key, image):
        bytes = self.image2bytes(image)
        lmdb_txn.put(key, bytes)

    def test(self, model, val_loader, tar_dir):

        if not os.path.isdir(tar_dir):
            os.makedirs(tar_dir)

        lmdb_env = lmdb.open(tar_dir, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True)

        for i, data in (enumerate(val_loader)):

            images_hr, images_lr, label_strs = data #, label_key_hr, label_key_lr

            # img_HR_key_new = b'image_hr-%09d' % cnt

            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)

            images_dr = model(images_hr)

            images_dr = images_dr * 255.
            images_dr[images_dr > 255.] = 255
            images_dr[images_dr < 0.] = 0.

            images_lr = images_lr * 255.
            images_hr = images_hr * 255.

            # [N, 4, 32, 128]
            images_dr_np = images_dr.data.cpu().numpy().astype(np.uint8)
            images_lr_np = images_lr.data.cpu().numpy().astype(np.uint8)
            images_hr_np = images_hr.data.cpu().numpy().astype(np.uint8)

            N = images_dr_np.shape[0]
            # [N, 32, 128, 4 -> 3]
            images_dr_np = np.transpose(images_dr_np, (0, 2, 3, 1))[..., :3]
            images_lr_np = np.transpose(images_lr_np, (0, 2, 3, 1))[..., :3]
            images_hr_np = np.transpose(images_hr_np, (0, 2, 3, 1))[..., :3]

            print("Batch_num:", i)

            for batch_idx in range(N):

                image_dr = Image.fromarray(images_dr_np[batch_idx])
                image_lr = Image.fromarray(images_lr_np[batch_idx])
                image_hr = Image.fromarray(images_hr_np[batch_idx])
                label = label_strs[batch_idx]

                # Start from 1
                id = i * self.batch_size + (batch_idx + 1)
                # print("id:", id, i, N)
                img_HR_key_new = b'image_hr-%09d' % id
                img_LR_key_new = b'image_lr-%09d' % id
                img_LRReal_key_new = b'image_lrreal-%09d' % id
                label_key_new = b'label-%09d' % id

                self.write2lmdb(lmdb_txn, img_HR_key_new, image_hr)
                self.write2lmdb(lmdb_txn, img_LR_key_new, image_dr)
                self.write2lmdb(lmdb_txn, img_LRReal_key_new, image_lr)
                lmdb_txn.put(label_key_new, str(label).encode())

            if _DEBUG:

                vis_dir = "image_" + self.resume.split("/")[-2]
                if not os.path.isdir(vis_dir):
                    os.makedirs(vis_dir)

                lr = cv2.resize(images_lr_np[0], (128, 32), interpolation=cv2.INTER_CUBIC)
                sr = cv2.resize(images_dr_np[0], (128, 32), interpolation=cv2.INTER_CUBIC)
                hr = cv2.resize(images_hr_np[0], (128, 32), interpolation=cv2.INTER_CUBIC)

                # prob_val_i = cv2.resize(prob_val_i, (128, 32), interpolation=cv2.INTER_CUBIC)
                # char_mask = cv2.resize(char_mask, (128, 32), interpolation=cv2.INTER_CUBIC)

                paddimg_im = np.zeros(
                    (sr.shape[0] + lr.shape[0] + hr.shape[0] + hr.shape[0] + hr.shape[0] + 25, sr.shape[1], 3))
                paddimg_im[:lr.shape[0], :lr.shape[1], :] = lr
                paddimg_im[lr.shape[0] + 5:lr.shape[0] + sr.shape[0] + 5, :sr.shape[1], :] = sr
                paddimg_im[lr.shape[0] + sr.shape[0] + 10:lr.shape[0] + sr.shape[0] + 10 + hr.shape[0],
                :hr.shape[1],
                :] = hr

                file_name = "" + str(i * 64) + ".jpg"
                cv2.imwrite(os.path.join(vis_dir, file_name),
                            cv2.cvtColor(paddimg_im.astype(np.uint8), cv2.COLOR_RGB2BGR))

        lmdb_txn.put(b'num-samples', str(id).encode())
        lmdb_txn.commit()

