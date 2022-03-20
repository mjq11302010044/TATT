import torch
import sys
import os
from tqdm import tqdm
import math
import torch.nn as nn
import torch.optim as optim
from IPython import embed
import math
import cv2
import string
from PIL import Image
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict

import ptflops

from model import tsrn, bicubic, srcnn, vdsr, srresnet, edsr, esrgan, rdn, lapsrn, transformerSR, scgan, tbsrn, han, pcan
from model import recognizer
from model import moran
from model import crnn
from dataset import lmdbDataset, \
    alignCollate_real, ConcatDataset, lmdbDataset_real, \
    alignCollate_syn, lmdbDataset_mix, alignCollateW2V_real, \
    lmdbDatasetWithW2V_real, alignCollatec2f_real, lmdbDataset_realIC15, \
    alignCollate_realWTL, alignCollate_realWTL_withcrop, alignCollate_realWTLAMask, \
    lmdbDatasetWithMask_real, lmdbDataset_realIC15TextSR, lmdbDataset_realCOCOText, \
    lmdbDataset_realSVT, lmdbDataset_realCHNSyn, lmdbDataset_realBadSet, lmdbDataset_CSVTR, \
    lmdbDataset_realDistorted, lmdbDataset_realBadSet
from loss import gradient_loss, percptual_loss, image_loss, semantic_loss

from utils.labelmaps import get_vocabulary, labels2strs

sys.path.append('../')
from utils import util, ssim_psnr, utils_moran, utils_crnn
import dataset.dataset as dataset


class TextBase(object):
    def __init__(self, config, args, opt_TPG=None):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        self.opt_TPG = opt_TPG

        if self.args.arch == "tsrn":
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = self.align_collate
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == "sem_tsrn":
            self.align_collate = alignCollateW2V_real
            self.load_dataset = lmdbDatasetWithW2V_real

            self.load_dataset_val = lmdbDatasetWithW2V_real

        elif self.args.arch == "tsrn_c2f":
            self.align_collate = alignCollatec2f_real
            self.load_dataset = lmdbDataset_real

        elif self.args.arch == "tsrn_tl":
            self.align_collate = alignCollate_realWTL
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == 'tsrn_tl_wmask':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real
        elif self.args.arch == 'tsrn_tl_cascade':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real#Distorted# BadSet

        elif self.args.arch == 'tsrn_tl_cascade_sft':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real#Distorted# BadSet

        elif self.args.arch == 'tatt':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real#Distorted# BadSet

        elif self.args.arch == 'tbsrn_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == 'tranSR_v4':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == 'srcnn_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == 'srresnet_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == 'rdn_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == 'vdsr_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real

        elif self.args.arch == 'esrgan_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real
        elif self.args.arch == 'scgan_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real
        elif self.args.arch == 'pcan_tl':
            self.align_collate = alignCollate_realWTLAMask
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = alignCollate_realWTL
            self.load_dataset_val = lmdbDataset_real
        else:

            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_real

            self.align_collate_val = self.align_collate
            self.load_dataset_val = lmdbDataset_realDistorted

        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation,
            'chinese': open("al_chinese.txt", "r").readlines()[0].replace("\n", "")
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.ckpt_path = os.path.join('ckpt', self.vis_dir)
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.cal_psnr_weighted = ssim_psnr.weighted_calculate_psnr
        self.cal_ssim_weighted = ssim_psnr.SSIM_WEIGHTED()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len,
                                      rotate=self.args.rotate_train,
                                      test=False
                ))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        # print("cfg.down_sample_scale:", cfg.down_sample_scale)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=True),
            drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        self.args.test_data_dir

        if self.args.go_test:
            test_dataset = self.load_dataset_val(root=dir_,
                                             voc_type=cfg.voc_type,
                                             max_len=cfg.max_len,
                                             test=True,
                                             rotate=self.args.rotate_test
                                             )
        else:
            test_dataset = self.load_dataset_val(root=dir_,  #load_dataset
                                             voc_type=cfg.voc_type,
                                             max_len=cfg.max_len,
                                             test=True,
                                                 rotate=self.args.rotate_test
                                             )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate_val(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask, train=False),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self, iter=-1, resume_in=None):
        cfg = self.config.TRAIN

        resume = self.resume
        if not resume_in is None:
            resume = resume_in

        if self.args.arch == 'tsrn':
            model = tsrn.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch == 'tsrn_c2f':
            model = tsrn.TSRN_C2F(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'sem_tsrn':
            model = tsrn.SEM_TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                              STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_loss_com = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
            semantic_loss_com = semantic_loss.SemanticLoss()
            image_crit = {"image_loss": image_loss_com, "semantic_loss": semantic_loss_com}
        elif self.args.arch == 'tsrn_tl':
            model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                  STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                  hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'tsrn_tl_wmask':
            model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                  STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                  hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'tsrn_tl_cascade':
            model = tsrn.TSRN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                 STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                 hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch in ['tatt', 'tatt_GlobalSR']:
            model = tsrn.TSRN_TL_TRANS(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                 STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                 hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch == "tsrn_tl_cascade_sft":
            model = tsrn.TSRN_TL_SFT(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb,
                                       hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'bicubic' and self.args.test:
            model = bicubic.BICUBIC(scale_factor=self.scale_factor)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srcnn':
            model = srcnn.SRCNN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'vdsr':
            model = vdsr.VDSR(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srres':
            model = srresnet.SRResNet(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'esrgan':
            model = esrgan.RRDBNet(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'scgan':
            model = scgan.SCGAN(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'rdn':
            model = rdn.RDN(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'edsr':
            model = edsr.EDSR(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'lapsrn':
            model = lapsrn.LapSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = lapsrn.L1_Charbonnier_loss()

        elif self.args.arch == 'srcnn_tl':
            model = srcnn.SRCNN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'han':
            model = han.HAN()
            image_crit = nn.MSELoss()
        elif self.args.arch == 'pcan':
            model = pcan.PCAN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.EdgeImageLoss()
        elif self.args.arch == 'srresnet_tl':
            model = srresnet.SRResNet_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'rdn_tl':
            model = rdn.RDN_TL(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'vdsr_tl':
            model = vdsr.VDSR_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == "pcan_tl":
            model = pcan.PCAN_TL(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch == "tranSR_v4":
            model = transformerSR.SRTransformer_V4()
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])

        elif self.args.arch == 'esrgan_tl':
            model = esrgan.RRDBNet_TL(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()

        elif self.args.arch == 'scgan_tl':
            model = scgan.SCGAN_TL(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        else:
            raise ValueError

        channel_size = 3 if self.args.arch in ["srcnn", "edsr", "vdsr", ""] else 4
        macs, params = ptflops.get_model_complexity_info(model, (channel_size, cfg.height//cfg.down_sample_scale, cfg.width//cfg.down_sample_scale), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- SR Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        if self.args.arch != 'bicubic':
            model = model.to(self.device)
            if self.args.arch == 'sem_tsrn':
                for k in image_crit.keys():
                    image_crit[k] = image_crit[k].to(self.device)
            else:
                image_crit.to(self.device)
            if cfg.ngpu > 1:

                print("multi_gpu", self.device)

                model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))

                if self.args.arch == 'sem_tsrn':
                    for k in image_crit.keys():
                        image_crit[k] = torch.nn.DataParallel(image_crit[k], device_ids=range(cfg.ngpu))
                else:
                    image_crit = torch.nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))

            if not resume == '':
                print('loading pre-trained model from %s ' % resume)
                if self.config.TRAIN.ngpu == 1:
                    # if is dir, we need to initialize the model list
                    if os.path.isdir(resume):
                        print("resume:", resume)
                        model_dict = torch.load(
                                os.path.join(resume, "model_best_acc_" + str(iter) + ".pth")
                            )['state_dict_G']
                        # print("model_dict:", model_dict.keys())

                        old_state_dict = model_dict
                        new_state_dict = model.state_dict()

                        # for key in old_state_dict:
                        #for new_key in new_state_dict:
                        #    old_keys = list(old_state_dict.keys())

                        #    if not new_key in old_keys:
                        #        print("new_key:", new_key)


                        model.load_state_dict(
                            model_dict
                        , strict=False)



                    else:
                        loaded_state_dict = torch.load(resume)
                        if 'state_dict_G' in loaded_state_dict:
                            model.load_state_dict(torch.load(resume)['state_dict_G'])
                        else:
                            model.load_state_dict(torch.load(resume))
                else:
                    model_dict = torch.load(
                        os.path.join(resume, "model_best_acc_" + str(iter) + ".pth")
                    )['state_dict_G']

                    if os.path.isdir(resume):
                        model.load_state_dict(
                            {'module.' + k: v for k, v in model_dict.items()}
                            , strict=False)
                    else:
                        model.load_state_dict(
                        {'module.' + k: v for k, v in torch.load(resume)['state_dict_G'].items()})
        return {'model': model, 'crit': image_crit}

    def global_model_init(self, iter=-1):

        cfg = self.config.TRAIN

        model = srresnet.SRResNet(scale_factor=1, width=cfg.width, height=cfg.height,
                                  STN=self.args.STN, mask=self.mask)
        image_crit = nn.MSELoss()

        if self.args.arch != 'bicubic':
            model = model.to(self.device)
            if self.args.arch == 'sem_tsrn':
                for k in image_crit.keys():
                    image_crit[k] = image_crit[k].to(self.device)
            else:
                image_crit.to(self.device)
            if cfg.ngpu > 1:

                model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))

                if self.args.arch == 'sem_tsrn':
                    for k in image_crit.keys():
                        image_crit[k] = torch.nn.DataParallel(image_crit[k], device_ids=range(cfg.ngpu))
                else:
                    image_crit = torch.nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))

            if not self.resume == '':
                print('loading pre-trained model from %s ' % self.resume)
                if self.config.TRAIN.ngpu == 1:
                    # if is dir, we need to initialize the model list
                    if os.path.isdir(self.resume):
                        model.load_state_dict(
                            torch.load(
                                os.path.join(self.resume, "global_model_best.pth")
                            )
                        )
                    else:
                        model.load_state_dict(torch.load(self.resume)['state_dict_G'])
                else:
                    if os.path.isdir(self.resume):
                        model.load_state_dict(
                            {'module.' + k: v for k, v in torch.load(
                                os.path.join(self.resume, "global_model_best.pth")
                            ).items()}
                        )
                    else:
                        model.load_state_dict(
                        {'module.' + k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})

        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model, recognizer=None, global_model=None):
        cfg = self.config.TRAIN

        # print("recognizer:", recognizer)

        if not recognizer is None:

            if type(recognizer) == list:
                if cfg.optimizer == "Adam":

                    rec_params = []
                    model_params = []

                    for recg in recognizer:
                        rec_params += list(recg.parameters())

                    if not global_model is None:
                        gm_params = []
                        if type(global_model) == list:
                            for gm in global_model:
                                gm_params += list(gm.parameters())
                        else:
                            gm_params += list(global_model.parameters())
                        model_params += gm_params

                    if type(model) == list:
                        for m in model:
                            model_params += list(m.parameters())
                    else:
                        model_params += list(model.parameters())

                    optimizer = optim.Adam(model_params + rec_params, lr=cfg.lr,
                                           betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "SGD":
                    optimizer = optim.SGD(list(model.parameters()) + rec_params, lr=cfg.lr,
                                          momentum=0.9)
            else:
                if cfg.optimizer == "Adam":

                    model_params = []
                    if type(model) == list:
                        for m in model:
                            model_params += list(m.parameters())
                    else:
                        model_params = list(model.parameters())

                    optimizer = optim.Adam(model_params + list(recognizer.parameters()), lr=cfg.lr,
                                           betas=(cfg.beta1, 0.999))
                elif cfg.optimizer == "SGD":
                    optimizer = optim.SGD(list(model.parameters()) + list(recognizer.parameters()), lr=cfg.lr,
                                          momentum=0.9)

        else:
            model_params = []
            if type(model) == list:
                for m in model:
                    model_params += list(m.parameters())
            else:
                model_params = list(model.parameters())

            if cfg.optimizer == "Adam":
                optimizer = optim.Adam(model_params, lr=cfg.lr,
                                       betas=(cfg.beta1, 0.999))
            elif cfg.optimizer == "SGD":
                optimizer = optim.SGD(model_params, lr=cfg.lr,
                                      momentum=0.9)

        return optimizer

    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(self.config.TRAIN.VAL.n_vis)):
            # embed()
            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            tensor_target = image_target[i][:3,:,:]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join('./demo', self.vis_dir)
            if not os.path.exists(out_root):
                os.mkdir(out_root)
            out_path = os.path.join(out_root, str(index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            if index is not 0:
                torchvision.utils.save_image(vis_im, os.path.join(out_path, im_name), padding=0)

    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, str_filt):
        visualized = 0
        for i in (range(image_in.shape[0])):
            if True:
                if (str_filt(pred_str_lr[i], 'lower') != str_filt(label_strs[i], 'lower')) and \
                        (str_filt(pred_str_sr[i], 'lower') == str_filt(label_strs[i], 'lower')):
                    visualized += 1
                    tensor_in = image_in[i].cpu()
                    tensor_out = image_out[i].cpu()
                    tensor_target = image_target[i].cpu()
                    transform = transforms.Compose(
                        [transforms.ToPILImage(),
                         transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                         transforms.ToTensor()]
                    )
                    tensor_in = transform(tensor_in)
                    images = ([tensor_in, tensor_out, tensor_target])
                    vis_im = torch.stack(images)
                    vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
                    out_root = os.path.join('./display', self.vis_dir)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
                    im_name = im_name.replace('/', '')
                    torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized

    def save_checkpoint(self, netG_list, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, recognizer=None, prefix="acc", global_model=None):

        ckpt_path = self.ckpt_path# = os.path.join('ckpt', self.vis_dir)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        for i in range(len(netG_list)):
            netG_ = netG_list[i]

            net_state_dict = None
            if self.config.TRAIN.ngpu > 1:
                netG = netG_.module
            else:
                netG = netG_

            save_dict = {
                'state_dict_G': netG.state_dict(),
                'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                         'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
                'best_history_res': best_acc_dict,
                'best_model_info': best_model_info,
                'param_num': sum([param.nelement() for param in netG.parameters()]),
                'converge': converge_list,
            }

            if is_best:
                torch.save(save_dict, os.path.join(ckpt_path, 'model_best_' + prefix + '_' + str(i) + '.pth'))
            else:
                torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))

        if is_best:
            # torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        rec_state_dict = recognizer[i].state_dict()
                        torch.save(rec_state_dict, os.path.join(ckpt_path, 'recognizer_best_' + prefix + '_' + str(i) + '.pth'))
                else:
                    rec_state_dict = recognizer.state_dict()
                    torch.save(rec_state_dict, os.path.join(ckpt_path, 'recognizer_best.pth'))
            if not global_model is None:
                torch.save(global_model, os.path.join(ckpt_path, 'global_model_best.pth'))
        else:
            # torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))
            if not recognizer is None:
                if type(recognizer) == list:
                    for i in range(len(recognizer)):
                        torch.save(recognizer[i].state_dict(), os.path.join(ckpt_path, 'recognizer_' + str(i) + '.pth'))
                else:
                    torch.save(recognizer.state_dict(), os.path.join(ckpt_path, 'recognizer.pth'))
            if not global_model is None:
                torch.save(global_model.state_dict(), os.path.join(ckpt_path, 'global_model.pth'))

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)

        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- TP Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        print("recognizer_path:", recognizer_path)

        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        # print("stat_dict:", stat_dict.keys())
        if recognizer_path is None:
            model.load_state_dict(stat_dict)
        else:
            # print("stat_dict:", stat_dict)
            # print("stat_dict:", type(stat_dict) == OrderedDict)
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        # model #.eval()
        # model.eval()
        return model, aster_info

    def CRNNRes18_init(self, recognizer_path=None, opt=None):
        model = crnn.CRNN_ResNet18(32, 1, 37, 256)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        stat_dict = torch.load(model_path)
        # print("stat_dict:", stat_dict.keys())
        if recognizer_path is None:
            if stat_dict == model.state_dict():
                model.load_state_dict(stat_dict)
        else:
            model = stat_dict
        # model #.eval()
        # model.eval()
        return model, aster_info

    def TPG_init(self, recognizer_path=None, opt=None):
        model = crnn.Model(opt)
        model = model.to(self.device)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = recognizer_path if not recognizer_path is None else opt.saved_model
        print('loading pretrained TPG model from %s' % model_path)
        stat_dict = torch.load(model_path)

        model_keys = model.state_dict().keys()
        #print("state_dict:", len(stat_dict))
        if type(stat_dict) == list:
            print("state_dict:", len(stat_dict))
            stat_dict = stat_dict[0]#.state_dict()
        #load_keys = stat_dict.keys()

        # print("recognizer_path:", recognizer_path)

        if recognizer_path is None:
            # model.load_state_dict(stat_dict)
            load_keys = stat_dict.keys()
            man_load_dict = model.state_dict()
            for key in stat_dict:
                if not key.replace("module.", "") in man_load_dict:
                    print("Key not match", key, key.replace("module.", ""))
                man_load_dict[key.replace("module.", "")] = stat_dict[key]
            model.load_state_dict(man_load_dict)
        else:
            #model = stat_dict
            model.load_state_dict(stat_dict)

        return model, aster_info

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):

        in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def parse_OPT_data(self, imgs_input_, ratio_keep=False):

        in_width = 512

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        print('load pred_trained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        aster.eval()
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
