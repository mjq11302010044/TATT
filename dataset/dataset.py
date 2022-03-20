#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string
import cv2
import os
import re

sys.path.append('../')
from utils import str_filt
from utils.labelmaps import get_vocabulary, labels2strs
from IPython import embed
from pyfasttext import FastText
random.seed(0)

from utils import utils_deblur
from utils import utils_sisr as sr
from utils import utils_image as util
import imgaug.augmenters as iaa


from scipy import io as sio
scale = 0.90
kernel = utils_deblur.fspecial('gaussian', 15, 1.)
noise_level_img = 0.


def rand_crop(im):
    w, h = im.size
    p1 = (random.uniform(0, w*(1-scale)), random.uniform(0, h*(1-scale)))
    p2 = (p1[0] + scale*w, p1[1] + scale*h)
    return im.crop(p1 + p2)


def central_crop(im):
    w, h = im.size
    p1 = (((1-scale)*w/2), (1-scale)*h/2)
    p2 = ((1+scale)*w/2, (1+scale)*h/2)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im

class lmdbDataset_realBadSet(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False, rotate=False):
        super(lmdbDataset_realBadSet, self).__init__()

        # root should be detailed by upper folder of images

        # anno_dir = os.path.join(root, "ANNOTATION")

        self.imlist = os.listdir(root)
        self.image_dir = root
        # self.impath_list = []
        # self.anno_list = []

        print("collect images from:", root)

        # mode = "train" if root.split("/")[-2] == "TRAIN" else "test"
        self.nSamples = len(self.imlist)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        imfile = self.imlist[index]
        image_path = os.path.join(self.image_dir, imfile)

        print("imfile:", imfile)

        word = imfile.split("_")[1] if len(imfile.split("_")) > 1 else ""

        if not os.path.isfile(image_path):
            print("File not found for", image_path)
            return self[index+1]
        try:
            img_HR = Image.open(image_path)
            img_lr = img_HR.copy()

            img_lr_np = np.array(img_lr).astype(np.uint8)
            img_lry = cv2.cvtColor(img_lr_np, cv2.COLOR_RGB2YUV)[..., 0]
            img_lry = Image.fromarray(img_lry)

            img_HR_np = np.array(img_HR).astype(np.uint8)
            img_HRy = cv2.cvtColor(img_HR_np, cv2.COLOR_RGB2YUV)[..., 0]
            img_HRy = Image.fromarray(img_HRy)

            if img_HR.size[0] < 2 or img_HR.size[1] < 2:
                print("img_HR:", img_HR.size)
                return self[(index + 1) % self.nSamples]
        except ValueError:
            print("File not found for", image_path)
            return self[(index + 1) % self.nSamples]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)
        # label_str = str_filt(word, self.voc_type)

        return img_HR, img_lr, img_HRy, img_lry, imfile


class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=31, test=True):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        self.max_len = max_len
        self.voc_type = voc_type

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())

        try:
            img = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
        except IOError or len(label) > self.max_len:
            return self[index + 1]

        label_str = str_filt(word, self.voc_type)
        return img, label_str




def get_Syn_800K_with_words(mode, dataset_dir, lang_seq=False):
    # if mode == 'train':
    #    image_dir = os.path.join(dataset_dir, 'image_9000/')
    # gt_dir = os.path.join(dataset_dir, 'txt_9000/')

    # ./ICPR_dataset/update_ICPR_text_train_part1_20180316/train_1000/
    # else:
    #    image_dir = os.path.join(dataset_dir, 'image_1000/')
    # gt_dir = os.path.join(dataset_dir, 'txt_1000/')

    word2vec_mat = '../selected_smaller_dic.mat'
    #mat_data = sio.loadmat(word2vec_mat)
    #all_words = mat_data['selected_vocab']
    #all_vecs = mat_data['selected_dict']

    #w2v_dict = {}
    #print('Building w2v dictionary...')
    #for i in range(len(all_words)):
    #    w2v_dict[all_words[i][0][0]] = all_vecs[i]
    #print('done')

    mat_file = os.path.join(dataset_dir, 'gt.mat')
    # print('mat_file:', mat_file)
    mat_f = sio.loadmat(mat_file)

    wordBBs = mat_f['wordBB'][0]
    txt_annos = mat_f['txt'][0]
    im_names = mat_f['imnames'][0]

    sam_size = len(txt_annos)

    # image_list = os.listdir(image_dir)
    # image_list.sort()
    im_infos = []

    if mode == 'train':
        cache_pkl = './data_cache/Syn_800K_training'
    else:
        cache_pkl = './data_cache/Syn_800K_testing'

    if lang_seq:
        cache_pkl += "_lang_seq"
    cache_pkl += "_E2E.pkl"

    if os.path.isfile(cache_pkl):
        return pickle.load(open(cache_pkl, 'rb'))

    pro_cnt = 0

    im_range = (0, 200000) if mode == "train" else (200000, 205000)

    for i in range(im_range[0], im_range[1]):
        txts = txt_annos[i]
        im_path = os.path.join(dataset_dir, im_names[i][0])
        word_boxes = wordBBs[i]

        pro_cnt += 1
        if pro_cnt % 2000 == 0:
            print('processed image:', str(pro_cnt) + '/' + str(im_range[1] - im_range[0]))

        cnt = 0
        # print('word_boxes:', word_boxes.shape)
        im = cv2.imread(im_path)

        if len(word_boxes.shape) < 3:
            word_boxes = np.expand_dims(word_boxes, -1)
        words = []
        boxes = []
        word_vecs = []

        for txt in txts:
            txtsp = txt.split('\n')
            for line in txtsp:
                line = line.replace('\n', '').replace('\n', '').replace('\r', '').replace('\t', '').split(' ')
                # print('line:', line)
                for w in line:
                    # w = w
                    if len(w) > 0:
                        gt_ind = np.transpose(np.array(word_boxes[:, :, cnt], dtype=np.int32), (1, 0)).reshape(8)
                        # print(imname, gt_ind, w)
                        cnt += 1
                        '''
                        cv2.line(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                        cv2.line(im, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 3)
                        cv2.line(im, (box[4], box[5]), (box[6], box[7]), (0, 0, 255), 3)
                        cv2.line(im, (box[6], box[7]), (box[0], box[1]), (0, 0, 255), 3)
                        cv2.putText(im, w, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 122), 2)
                        '''

                        pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                        pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                        pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                        pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                        angle = 0

                        if edge1 > edge2:

                            width = edge1
                            height = edge2
                            if pt1[0] - pt2[0] != 0:
                                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        elif edge2 >= edge1:
                            width = edge2
                            height = edge1
                            # print pt2[0], pt3[0]
                            if pt2[0] - pt3[0] != 0:
                                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        if angle < -45.0:
                            angle = angle + 180

                        x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                        y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                        if height * width * (800 / float(im.shape[0])) < 16 * 32 and mode == "train":
                            continue
                        if x_ctr >= im.shape[1] or x_ctr < 0 or y_ctr >= im.shape[0] or y_ctr < 0:
                            continue

                        #com_num = re.compile('[0-9]+')
                        #com_prices = re.compile('[$￥€£]+')

                        #match_num = re.findall(com_num, w)
                        #match_prices = re.findall(com_prices, w)

                        # choices: original, prices, others
                        # 2 for English
                        if lang_seq:
                            w = ["1" for i in range(len(w))]
                            w = "".join(w)
                        words.append(w)
                        '''
                        w = w.lower()
                        if w in w2v_dict:
                            word_vecs.append(w2v_dict[w.lower()])
                        elif match_prices and match_num:
                            word_vecs.append(w2v_dict['price'])
                        elif match_num and not match_prices:
                            word_vecs.append(w2v_dict['ten'])
                        else:
                            print(im_path, w)
                            word_vecs.append(np.zeros(100, dtype=np.float32) + 1e-10)
                        '''

                        gt_ptx = gt_ind.reshape(-1, 2)

                        xmax = np.max(gt_ptx[:, 0])
                        xmin = np.min(gt_ptx[:, 0])
                        ymax = np.max(gt_ptx[:, 1])
                        ymin = np.min(gt_ptx[:, 1])

                        # return to width, height
                        boxes.append([xmin, ymin, xmax - xmin, ymax - ymin]) #x_ctr, y_ctr, width, height, angle, w
        cls_num = 2
        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 4), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # prob
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]] #, boxes[idx][4]

        # print ("boxes_size:", gt_boxes.shape[0])
        if gt_boxes.shape[0] > 0:
            max_overlaps = overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = overlaps.argmax(axis=1)
        else:
            continue

        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': im_path,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'gt_words': words,
            # 'gt_wordvec': np.array(word_vecs),
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_pkl, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print("Save pickle done.")
    return im_infos



class lmdbDataset_GlobalSR(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=31, test=False, rotate=False):
        super(lmdbDataset_GlobalSR, self).__init__()

        if test:
            mode = "test"
        else:
            mode = "train"

        self.image_dataset = get_Syn_800K_with_words(mode, dataset_dir=root, lang_seq=False)
        self.nSamples = len(self.image_dataset)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # index += 1
        '''
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())

        try:
            img = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
        except IOError or len(label) > self.max_len:
            return self[index + 1]

        label_str = str_filt(word, self.voc_type)
        '''
        image_info = self.image_dataset[index]
        impath = image_info['image']
        image_pil = Image.open(impath)
        boxes = image_info['boxes']
        gt_words = image_info['gt_words']

        return image_pil, boxes, gt_words



def gauss_unsharp_mask(rgb, shp_kernel, shp_sigma, shp_gain):
    LF = cv2.GaussianBlur(rgb, (shp_kernel, shp_kernel), shp_sigma)
    HF = rgb - LF
    RGB_peak = rgb + HF * shp_gain
    RGB_noise_NR_shp = np.clip(RGB_peak, 0.0, 255.0)
    return RGB_noise_NR_shp, LF


def add_shot_gauss_noise(rgb, shot_noise_mean, read_noise):
    noise_var_map = shot_noise_mean * rgb + read_noise
    noise_dev_map = np.sqrt(noise_var_map)
    noise = np.random.normal(loc=0.0, scale = noise_dev_map, size=None)
    if (rgb.mean() > 252.0):
        noise_rgb = rgb
    else:
        noise_rgb = rgb + noise
    noise_rgb = np.clip(noise_rgb, 0.0, 255.0)
    return noise_rgb


def degradation(src_img):
    # RGB Image input
    GT_RGB = np.array(src_img)
    GT_RGB = GT_RGB.astype(np.float32)

    pre_blur_kernel_set = [3, 5]
    sharp_kernel_set = [3, 5]
    blur_kernel_set = [5, 7, 9, 11]
    NR_kernel_set = [3, 5]

    # Pre Blur
    kernel = pre_blur_kernel_set[random.randint(0, (len(pre_blur_kernel_set) - 1))]
    blur_sigma = random.uniform(5., 6.)
    RGB_pre_blur = cv2.GaussianBlur(GT_RGB, (kernel, kernel), blur_sigma)

    rand_p = random.random()
    if rand_p > 0.2:
        # Noise
        shot_noise = random.uniform(0, 0.005)
        read_noise = random.uniform(0, 0.015)
        GT_RGB_noise = add_shot_gauss_noise(RGB_pre_blur, shot_noise, read_noise)
    else:
        GT_RGB_noise = RGB_pre_blur

    # Noise Reduction
    choice = random.uniform(0, 1.0)
    GT_RGB_noise = np.round(GT_RGB_noise)
    GT_RGB_noise = GT_RGB_noise.astype(np.uint8)
    # if (shot_noise < 0.06):
    if (choice < 0.7):
        NR_kernel = NR_kernel_set[random.randint(0, (len(NR_kernel_set) - 1))]  ###3,5,7,9
        NR_sigma = random.uniform(2., 3.)
        GT_RGB_noise_NR = cv2.GaussianBlur(GT_RGB_noise, (NR_kernel, NR_kernel), NR_sigma)
    else:
        value_sigma = random.uniform(70, 80)
        space_sigma = random.uniform(70, 80)
        GT_RGB_noise_NR = cv2.bilateralFilter(GT_RGB_noise, 7, value_sigma, space_sigma)

    # Sharpening
    GT_RGB_noise_NR = GT_RGB_noise_NR.astype(np.float32)
    shp_kernel = sharp_kernel_set[random.randint(0, (len(sharp_kernel_set) - 1))]  ###5,7,9
    shp_sigma = random.uniform(2., 3.)
    shp_gain = random.uniform(3., 4.)
    RGB_noise_NR_shp, LF = gauss_unsharp_mask(GT_RGB_noise_NR, shp_kernel, shp_sigma, shp_gain)

    # print("RGB_noise_NR_shp:", RGB_noise_NR_shp.shape)

    return Image.fromarray(RGB_noise_NR_shp.astype(np.uint8))


def noisy(noise_typ,image):

    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 50
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        # print("gauss:", np.unique(gauss))
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def JPEG_compress(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    ret_img = cv2.imdecode(encimg, 1)
    return ret_img

class lmdbDataset_real(Dataset):
    def __init__(
                 self, root=None,
                 voc_type='upper',
                 max_len=100,
                 test=False,
                 cutblur=False,
                 manmade_degrade=False,
                 rotate=None
                 ):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.cb_flag = cutblur
        self.rotate = rotate

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("nSamples:", nSamples)
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        self.manmade_degrade = manmade_degrade

    def __len__(self):
        return self.nSamples

    def rotate_img(self, image, angle):
        # convert to cv2 image

        if not angle == 0.0:
            image = np.array(image)
            (h, w) = image.shape[:2]
            scale = 1.0
            # set the rotation center
            center = (w / 2, h / 2)
            # anti-clockwise angle in the function
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, M, (w, h))
            # back to PIL image
            image = Image.fromarray(image)
            
        return image


    def cutblur(self, img_hr, img_lr):
        p = random.random()

        img_hr_np = np.array(img_hr)
        img_lr_np = np.array(img_lr)

        randx = int(img_hr_np.shape[1] * (0.2 + 0.8 * random.random()))

        if p > 0.7:
            left_mix = random.random()
            if left_mix <= 0.5:
                img_lr_np[:, randx:] = img_hr_np[:, randx:]
            else:
                img_lr_np[:, :randx] = img_hr_np[:, :randx]

        return Image.fromarray(img_lr_np)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = ""#str(txn.get(label_key).decode())
        # print("in dataset....")
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            if self.manmade_degrade:
                img_lr = degradation(img_HR)
            else:
                img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            # print("GOGOOGO..............", img_HR.size)
            if self.cb_flag and not self.test:
                img_lr = self.cutblur(img_HR, img_lr)

            if not self.rotate is None:

                if not self.test:
                    angle = random.random() * self.rotate * 2 - self.rotate
                else:
                    angle = 0 #self.rotate

                # img_HR = self.rotate_img(img_HR, angle)
                # img_lr = self.rotate_img(img_lr, angle)

            img_lr_np = np.array(img_lr).astype(np.uint8)
            img_lry = cv2.cvtColor(img_lr_np, cv2.COLOR_RGB2YUV)
            img_lry = Image.fromarray(img_lry)

            img_HR_np = np.array(img_HR).astype(np.uint8)
            img_HRy = cv2.cvtColor(img_HR_np, cv2.COLOR_RGB2YUV)
            img_HRy = Image.fromarray(img_HRy)
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())
            # print("img_HR:", img_HR.size, img_lr.size())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, img_HRy, img_lry, label_str


class lmdbDataset_realDistorted(Dataset):
    def __init__(
            self, root=None,
            voc_type='upper',
            max_len=100,
            test=False,
            cutblur=False,
            manmade_degrade=False,
            rotate=None
    ):
        super(lmdbDataset_realDistorted, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.cb_flag = cutblur
        self.rotate = rotate

        self.split = root.split("/")[-1]

        self.picked_index = open(os.path.join('./datasets/', self.split + "_distorted.txt"), "r").readlines()
        self.picked_index = [int(index) for index in self.picked_index if len(index) > 0]

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            self.nSamples = len(self.picked_index)
            print("nSamples:", self.nSamples)
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        self.manmade_degrade = manmade_degrade

    def __len__(self):
        return self.nSamples

    def rotate_img(self, image, angle):
        # convert to cv2 image

        if not angle == 0.0:
            image = np.array(image)
            (h, w) = image.shape[:2]
            scale = 1.0
            # set the rotation center
            center = (w / 2, h / 2)
            # anti-clockwise angle in the function
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, M, (w, h))
            # back to PIL image
            image = Image.fromarray(image)

        return image

    def cutblur(self, img_hr, img_lr):
        p = random.random()

        img_hr_np = np.array(img_hr)
        img_lr_np = np.array(img_lr)

        randx = int(img_hr_np.shape[1] * (0.2 + 0.8 * random.random()))

        if p > 0.7:
            left_mix = random.random()
            if left_mix <= 0.5:
                img_lr_np[:, randx:] = img_hr_np[:, randx:]
            else:
                img_lr_np[:, :randx] = img_hr_np[:, :randx]

        return Image.fromarray(img_lr_np)

    def __getitem__(self, index_):
        assert index_ <= len(self), 'index range error'
        # index += 1
        #####################################
        index = self.picked_index[index_]
        #####################################
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = ""  # str(txn.get(label_key).decode())
        # print("in dataset....")
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            if self.manmade_degrade:
                img_lr = degradation(img_HR)
            else:
                img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            # print("GOGOOGO..............", img_HR.size)
            if self.cb_flag and not self.test:
                img_lr = self.cutblur(img_HR, img_lr)

            if not self.rotate is None:

                if not self.test:
                    angle = random.random() * self.rotate * 2 - self.rotate
                else:
                    angle = 0  # self.rotate

                # img_HR = self.rotate_img(img_HR, angle)
                # img_lr = self.rotate_img(img_lr, angle)

            img_lr_np = np.array(img_lr).astype(np.uint8)
            img_lry = cv2.cvtColor(img_lr_np, cv2.COLOR_RGB2YUV)
            img_lry = Image.fromarray(img_lry)

            img_HR_np = np.array(img_HR).astype(np.uint8)
            img_HRy = cv2.cvtColor(img_HR_np, cv2.COLOR_RGB2YUV)
            img_HRy = Image.fromarray(img_HRy)
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())
            # print("img_HR:", img_HR.size, img_lr.size())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, img_HRy, img_lry, label_str

import pickle
class lmdbDataset_realCHNSyn(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realCHNSyn, self).__init__()

        flist = os.listdir(root)
        self.root_dir = root

        self.database_dict = {}

        print("Loading pkl files from", root, "...")
        for f in flist:
            if f.endswith(".pkl"):
                print("f:", f)
                with open(os.path.join(root, f), "rb") as pkl_f:
                    self.database_dict.update(pickle.load(pkl_f))

        self.nSamples = len(self.database_dict.keys())
        self.keys = list(self.database_dict.keys())
        print("done")
        print("All data:", self.nSamples)

        self.voc_type = voc_type

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        imkey = self.keys[index % self.nSamples]
        impath = os.path.join(self.root_dir, imkey + ".jpg")
        word = self.database_dict[imkey]

        try:
            img_HR = Image.open(impath)
            img_lr = img_HR.copy()

            img_lr_np = np.array(img_lr).astype(np.uint8)
            img_lr_np = cv2.GaussianBlur(img_lr_np, (5, 5), 1)
            img_lr = Image.fromarray(img_lr_np)

            img_lry = cv2.cvtColor(img_lr_np, cv2.COLOR_RGB2YUV)[..., 0]
            img_lry = Image.fromarray(img_lry)

            img_HR_np = np.array(img_HR).astype(np.uint8)
            img_HRy = cv2.cvtColor(img_HR_np, cv2.COLOR_RGB2YUV)[..., 0]
            img_HRy = Image.fromarray(img_HRy)
            # print("img_HR:", img_HR.size, img_lr.size())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, img_HRy, img_lry, label_str #


class lmdbDataset_realIC15TextSR(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realIC15TextSR, self).__init__()

        # root should be detailed by upper folder of images
        hr_image_dir = os.path.join(root, "HR")
        lr_image_dir = os.path.join(root, "LR")
        anno_dir = os.path.join(root, "ANNOTATION")

        hr_image_list = os.listdir(hr_image_dir)

        self.hr_impath_list = []
        self.lr_impath_list = []
        self.anno_list = []

        print("collect images from:", root)

        mode = "train" if root.split("/")[-2] == "TRAIN" else "test"

        for i in range(len(hr_image_list)):
            hr_impath = os.path.join(hr_image_dir, mode + '-hr-' + str(i+1).rjust(4, '0') + ".pgm")
            lr_impath = os.path.join(lr_image_dir, mode + '-lr-' + str(i+1).rjust(4, '0') + ".pgm")
            anno_path = os.path.join(anno_dir, mode + '-annot-' + str(i+1).rjust(4, '0') + ".txt")

            self.hr_impath_list.append(hr_impath)
            self.lr_impath_list.append(lr_impath)
            self.anno_list.append(anno_path)

        self.nSamples = len(self.anno_list)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def read_pgm(self, filename, byteorder='>'):
        """Return image data from a raw PGM file as numpy array.

        Format specification: http://netpbm.sourceforge.net/doc/pgm.html

        """
        with open(filename, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

            return np.frombuffer(buffer,
                                 dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                                 count=int(width) * int(height),
                                 offset=len(header)
                                 ).reshape((int(height), int(width)))

        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        # assert index <= len(self), 'index range error'

        if not os.path.isfile(self.hr_impath_list[idx]):
            print("File not found for", self.hr_impath_list[idx])
            return self[index+1]
        try:
            img_HR_np = self.read_pgm(self.hr_impath_list[idx], byteorder='<')
            img_lr_np = self.read_pgm(self.lr_impath_list[idx], byteorder='<')

            label_str = open(self.anno_list[idx], "r").readlines()[0].replace("\n", "").strip()
            label_str = str_filt(label_str, self.voc_type)
        except ValueError:
            print("File not found for", self.hr_impath_list[idx])
            return self[index + 1]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)

        img_HR = Image.fromarray(cv2.cvtColor(img_HR_np, cv2.COLOR_GRAY2RGB))
        img_lr = Image.fromarray(cv2.cvtColor(img_lr_np, cv2.COLOR_GRAY2RGB))

        return img_HR, img_lr, label_str



class lmdbDataset_realSVT(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realSVT, self).__init__()

        # root should be detailed by upper folder of images

        # anno_dir = os.path.join(root, "ANNOTATION")

        split = ("svt_" + "train") if not test else ("svt_" + "test")
        dataset_dir = os.path.join(root, split)
        self.image_dir = os.path.join(dataset_dir, "IMG")
        self.anno_dir = os.path.join(dataset_dir, "label")
        # self.impath_list = os.listdir(image_dir)
        self.anno_list = os.listdir(self.anno_dir)

        # self.impath_list = []
        # self.anno_list = []

        print("collect images from:", root)

        # mode = "train" if root.split("/")[-2] == "TRAIN" else "test"
        self.nSamples = len(self.anno_list)
        print("Done, we have ", self.nSamples, "samples...")

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        idx = index % self.nSamples

        anno = self.anno_list[index]
        image_path = os.path.join(self.image_dir, anno.split(".")[0] + ".jpg")
        anno_path = os.path.join(self.anno_dir, anno)

        if not os.path.isfile(image_path):
            print("File not found for", image_path)
            return self[index+1]
        try:
            word = open(anno_path, "r").readlines()[0].replace("\n", "")
            img_HR = Image.open(image_path)
            img_lr = img_HR
        except ValueError:
            print("File not found for", image_path)
            return self[index + 1]
        # print("annos:", img_HR_np.shape, img_lr_np.shape)
        label_str = str_filt(word, self.voc_type)

        return img_HR, img_lr, label_str


class lmdbDataset_realIC15(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False, rotate=None):
        super(lmdbDataset_realIC15, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.degrade = True

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        '''
        if not self.degrade:
            valid_cnt = 0
            for index in range(1, self.nSamples + 1):
                txn = self.env.begin(write=False)
                label_key = b'label-%09d' % index
                word = str(txn.get(label_key).decode())
                img_key = b'image-%09d' % index  # 128*32
                # img_lr_key = b'image_lr-%09d' % index  # 64*16
                # try:
                img_HR = buf2PIL(txn, img_key, 'RGB')
                img_lr_np = np.array(img_HR).astype(np.uint8)

                H, W = img_lr_np.shape[:2]
                if H * W < 1024:
                    valid_cnt += 1
            self.nSamples = valid_cnt
        '''
        print("We have", self.nSamples, "samples from", root)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        index = index % (self.nSamples+1)

        # print(self.nSamples, index)

        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_key = b'image-%09d' % index  # 128*32
        # img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_key, 'RGB')

            img_lr = img_HR

            img_lr_np = np.array(img_lr).astype(np.uint8)
            # print("img_lr_np:", img_lr_np.shape)

            if self.degrade:
                # img_lr_np = cv2.GaussianBlur(img_lr_np, (5, 5), 1)
                # shot_noise = random.uniform(0, 0.005)
                # read_noise = random.uniform(0, 0.015)
                # img_lr_np = add_shot_gauss_noise(img_lr_np, shot_noise, read_noise).astype(np.uint8)
                pass
                # print("img_lr_np:", img_lr_np.shape)
            else:
                if img_lr_np.shape[0] * img_lr_np.shape[1] > 1024:
                    return self[(index + 1) % self.nSamples]

            img_lr = Image.fromarray(img_lr_np)

            if img_lr.size[0] < 4 or img_lr.size[1] < 4:
                return self[index + 1]

            # print("img:", img_HR.size, word)
            # img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]

        # if img_HR.size[0] < 4 or img_HR.size[1] < 4:
        #     return self[index + 1]

        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, img_HR, img_lr, label_str

class lmdbDataset_CSVTR(Dataset):
    def __init__(self, root=None, voc_type='chinese', max_len=100, test=False):
        super(lmdbDataset_CSVTR, self).__init__()
        self.image_path_list = []

        self.imdir = os.path.join(root, "filter_dir")
        self.gt_file = os.path.join(root, "filter_train_test.list")

        self.gt_pairs = []

        gt_lines = open(self.gt_file, "r").readlines()
        for line in gt_lines:
            items = line.replace("\n", "").split("\t")
            self.gt_pairs.append([os.path.join(self.imdir, items[2]), items[3]])

        self.nSamples = len(self.gt_pairs)

        print("nSamples test:", self.nSamples)

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        word = self.gt_pairs[index][1]
        # print("word:", word)
        try:
            img_HR = Image.open(self.gt_pairs[index][0])  # for color image
            img_lr = Image.open(self.gt_pairs[index][0])

        except IOError:
            return self[index+1]

        #label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, img_HR, img_lr, word



class lmdbDataset_realCOCOText(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_realCOCOText, self).__init__()

        if test:
            gt_file = "val_words_gt.txt"
            im_dir = "val_words"
        else:
            gt_file = "train_words_gt.txt"
            im_dir = "train_words"

        self.image_dir = os.path.join(root, im_dir)
        self.gt_file = os.path.join(root, gt_file)

        self.gtlist = open(self.gt_file, "r").readlines()

        if test:
            self.gtlist = self.gtlist[:3000]

        self.nSamples = len(self.gtlist)

        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # index += 1

        gt_anno = self.gtlist[index].replace("\n", "")
        if len(gt_anno.split(",")) < 2:
            return self[index + 1]
        img_id, label_str = gt_anno.split(",")[:2]
        impath = os.path.join(self.image_dir, img_id + ".jpg")

        try:
            img_HR = Image.open(impath)
            img_lr = img_HR
            # print("img:", img_HR.size, word)
            # img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(label_str) > self.max_len:
            return self[index + 1]
        label_str = str_filt(label_str, self.voc_type)
        return img_HR, img_lr, label_str


class lmdbDatasetWithW2V_real(Dataset):
    def __init__(
                     self,
                     root=None,
                     voc_type='upper',
                     max_len=100,
                     test=False,
                     w2v_lexicons="cc.en.300.bin"
                 ):
        super(lmdbDatasetWithW2V_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        # self.w2v_lexicon = FastText(w2v_lexicons)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        # print("HR, LR:", img_HR.size, img_lr.size)

        w2v = None# self.w2v_lexicon.get_numpy_vector(label_str.lower())

        return img_HR, img_lr, label_str, w2v



class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug=None, blur=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
        self.aug = aug

        self.blur = blur

    def __call__(self, img, ratio_keep=False):

        size = self.size

        if ratio_keep:
            ori_width, ori_height = img.size
            ratio = float(ori_width) / ori_height

            if ratio < 3:
                width = 100# if self.size[0] == 32 else 50
            else:
                width = int(ratio * self.size[1])

            size = (width, self.size[1])

        # print("size:", size)
        img = img.resize(size, self.interpolation)

        if self.blur:
            # img_np = np.array(img)
            # img_np = cv2.GaussianBlur(img_np, (5, 5), 1)
            #print("in degrade:", np.unique(img_np))
            # img_np = noisy("gauss", img_np).astype(np.uint8)
            # img_np = apply_brightness_contrast(img_np, 40, 40).astype(np.uint8)
            # img_np = JPEG_compress(img_np)

            # img = Image.fromarray(img_np)
            pass

        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])

        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class NormalizeOnly(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug=None, blur=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
        self.aug = aug

        self.blur = blur

    def __call__(self, img, ratio_keep=False):

        size = self.size

        if ratio_keep:
            ori_width, ori_height = img.size
            ratio = float(ori_width) / ori_height

            if ratio < 3:
                width = 100# if self.size[0] == 32 else 50
            else:
                width = int(ratio * self.size[1])

            size = (width, self.size[1])

        # print("size:", size)
        # img = img.resize(size, self.interpolation)

        if self.blur:
            img_np = np.array(img)
            # img_np = cv2.GaussianBlur(img_np, (5, 5), 1)
            #print("in degrade:", np.unique(img_np))
            # img_np = noisy("gauss", img_np).astype(np.uint8)
            # img_np = apply_brightness_contrast(img_np, 40, 40).astype(np.uint8)
            # img_np = JPEG_compress(img_np)

            img = Image.fromarray(img_np)

        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])

        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor



class resizeNormalizeRandomCrop(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img, interval=None):

        w, h = img.size

        if w < 32 or not interval is None:
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)
        else:
            np_img = np.array(img)
            h, w = np_img.shape[:2]
            np_img_crop = np_img[:, int(w * interval[0]):int(w * interval[1])]
            # print("size:", self.size, np_img_crop.shape, np_img.shape, interval)
            img = Image.fromarray(np_img_crop)
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img)

        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class resizeNormalizeKeepRatio(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask

    def __call__(self, img, label_str):
        o_w, o_h = img.size

        ratio = o_w / float(o_h)
        re_h = self.size[1]
        re_w = int(re_h * ratio)
        if re_w > self.size[0]:
            img = img.resize(self.size, self.interpolation)
            img_tensor = self.toTensor(img).float()
        else:
            img = img.resize((re_w, re_h), self.interpolation)
            img_np = np.array(img)
            # if len(label_str) > 4:
            #     print("img_np:", img_np.shape)

            shift_w = int((self.size[0] - img_np.shape[1]) / 2)
            re_img = np.zeros((self.size[1], self.size[0], img_np.shape[-1]))
            re_img[:, shift_w:img_np.shape[1]+shift_w] = img_np

            re_img = Image.fromarray(re_img.astype(np.uint8))

            img_tensor = self.toTensor(re_img).float()

            if o_h / o_w < 0.5 and len(label_str) > 4:
                # cv2.imwrite("mask_h_" + label_str + ".jpg", re_mask.astype(np.uint8))
                # cv2.imwrite("img_h_" + label_str + ".jpg", np.array(re_img))
                # print("img_np_h:", o_h, o_w, img_np.shape, label_str)
                pass

        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            if re_w > self.size[0]:
                # img = img.resize(self.size, self.interpolation)

                re_mask_cpy = np.ones((mask.size[1], mask.size[0]))

                mask = self.toTensor(mask)
                img_tensor = torch.cat((img_tensor, mask), 0).float()
            else:
                mask = np.array(mask)
                mask = cv2.resize(mask, (re_w, re_h), cv2.INTER_NEAREST)
                shift_w = int((self.size[0] - mask.shape[1]) / 2)

                # print("resize mask:", mask.shape)

                re_mask = np.zeros((self.size[1], self.size[0]))

                re_mask_cpy = re_mask.copy()
                re_mask_cpy[:, shift_w:mask.shape[1] + shift_w] = np.ones(mask.shape)

                re_mask[:, shift_w:mask.shape[1] + shift_w] = mask
                '''
                if o_h / o_w > 2 and len(label_str) > 4:
                    cv2.imwrite("mask_" + label_str + ".jpg", re_mask.astype(np.uint8))
                    cv2.imwrite("img_" + label_str + ".jpg", re_img.astype(np.uint8))
                    print("img_np:", o_h, o_w, img_np.shape, label_str)

                if o_h / o_w < 0.5 and len(label_str) > 4:
                    cv2.imwrite("mask_h_" + label_str + ".jpg", re_mask.astype(np.uint8))
                    cv2.imwrite("img_h_" + label_str + ".jpg", re_img.astype(np.uint8))
                    print("img_np_h:", o_h, o_w, img_np.shape, label_str)
                '''
                re_mask = self.toTensor(re_mask).float()
                img_tensor = torch.cat((img_tensor, re_mask), 0)

        return img_tensor, torch.tensor(cv2.resize(re_mask_cpy, (self.size[0] * 2, self.size[1] * 2), cv2.INTER_NEAREST)).float()


class lmdbDataset_mix(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_mix, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        if self.test:
            try:
                img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            except:
                img_HR = buf2PIL(txn, b'image-%09d' % index, 'RGB')
                img_lr = img_HR

        else:
            img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
            if random.uniform(0, 1) < 0.5:
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            else:
                img_lr = img_HR

        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class lmdbDatasetWithMask_real(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDatasetWithMask_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def get_mask(self, image):

        img_hr = np.array(image)
        img_hr_gray = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), np.uint8)
        hr_canny = cv2.Canny(img_hr_gray, 20, 150)
        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        hr_canny = cv2.GaussianBlur(hr_canny, (5, 5), 1)
        weighted_mask = 0.4 + (hr_canny / 255.0) * 0.5

        return weighted_mask

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        weighted_mask = self.get_mask(img_HR)

        return img_HR, img_lr, label_str, weighted_mask



class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples



class alignCollate_syn(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True,
                 y_domain=False
                 ):

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)

        aug = [
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(1, 5)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(
                d=(3, 9), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.MotionBlur(k=3),
            iaa.MeanShiftBlur(),
            iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(1, 7))
        ]

        self.aug = iaa.Sequential([sometimes(a) for a in aug], random_order=True)

        # self.y_domain = y_domain

        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        # self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.alphabet = open("al_chinese.txt", "r").readlines()[0].replace("\n", "")
        self.d2a = "-" + self.alphabet
        self.alsize = len(self.d2a)
        self.a2d = {}
        cnt = 0
        for ch in self.d2a:
            self.a2d[ch] = cnt
            cnt += 1

        imgH = self.imgH
        imgW = self.imgW

        self.transform = resizeNormalize((imgW, imgH), self.mask)
        self.transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask, blur=True)
        self.transform_pseudoLR = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask, aug=self.aug)

        self.train = train

    def degradation(self, img_L):
        # degradation process, blur + bicubic downsampling + Gaussian noise
        # if need_degradation:
        # img_L = util.modcrop(img_L, sf)
        img_L = np.array(img_L)
        # print("img_L_before:", img_L.shape, np.unique(img_L))
        img_L = sr.srmd_degradation(img_L, kernel)

        noise_level_img = 0.
        if not self.train:
            np.random.seed(seed=0)  # for reproducibility
        # print("unique:", np.unique(img_L))
        img_L = img_L + np.random.normal(0, noise_level_img, img_L.shape)

        # print("img_L_after:", img_L_beore.shape, img_L.shape, np.unique(img_L))

        return Image.fromarray(img_L.astype(np.uint8))

    def __call__(self, batch):
        images, images_lr, _, _, label_strs = zip(*batch)

        # [self.degradation(image) for image in images]
        # images_hr = images
        '''
        images_lr = [image.resize(
            (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            Image.BICUBIC) for image in images]

        if self.train:
            if random.random() > 1.5:
                images_hr = [image.resize(
                (image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale),
                Image.BICUBIC) for image in images]
            else:
                images_hr = images
        else:
            images_hr = images
            #[image.resize(
            #    (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            #    Image.BICUBIC) for image in images]
        '''
        # images_hr = [self.degradation(image) for image in images]
        images_hr = images
        #images_lr = [image.resize(
        #     (image.size[0] // 4, image.size[1] // 4),
        #     Image.BICUBIC) for image in images_lr]
        # images_lr = images

        #images_lr_new = []
        #for image in images_lr:
        #    image_np = np.array(image)
        #    image_aug = self.aug(images=image_np[None, ])[0]
        #    images_lr_new.append(Image.fromarray(image_aug))
        #images_lr = images_lr_new

        images_hr = [self.transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        if self.train:
            images_lr = [image.resize(
            (image.size[0] // 2, image.size[1] // 2), # self.down_sample_scale
            Image.BICUBIC) for image in images_lr]
        else:
            pass
        #    # for image in images_lr:
        #    #     print("images_lr:", image.size)
        #    images_lr = [image.resize(
        #         (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),  # self.down_sample_scale
        #        Image.BICUBIC) for image in images_lr]
        #    pass
        # images_lr = [self.degradation(image) for image in images]
        images_lr = [self.transform2(image) for image in images_lr]

        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        max_len = 26

        label_batches = []
        weighted_tics = []
        weighted_masks = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                #inter_com = 26 - len(word)
                #padding = int(inter_com / (len(word) - 1))
                #new_word = word[0]
                #for i in range(len(word) - 1):
                #    new_word += "-" * padding + word[i + 1]

                #word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            if len(label_list) <= 0:
                # blank label
                weighted_masks.append(0)
            else:
                weighted_masks.extend(label_list)

            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            #if labels.shape[0] > 0:
            #    label_batches.append(label_vecs.scatter_(-1, labels, 1))
            #else:
            #    label_batches.append(label_vecs)

            if labels.shape[0] > 0:
                label_vecs = torch.zeros((labels.shape[0], self.alsize))
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
                weighted_tics.append(1)
            else:
                label_vecs = torch.zeros((1, self.alsize))
                label_vecs[0, 0] = 1.
                label_batches.append(label_vecs)
                weighted_tics.append(0)

        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        # print(images_lr.shape, images_hr.shape)

        return images_hr, images_lr, images_hr, images_lr, label_strs, label_rebatches, torch.tensor(weighted_masks).long(), torch.tensor(weighted_tics)


class alignCollate_syn_withcrop(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True
                 ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.d2a = "-" + self.alphabet
        self.alsize = len(self.d2a)
        self.a2d = {}
        cnt = 0
        for ch in self.d2a:
            self.a2d[ch] = cnt
            cnt += 1

        imgH = self.imgH
        imgW = self.imgW

        self.transform = resizeNormalizeRandomCrop((imgW, imgH), self.mask)
        self.transform2 = resizeNormalizeRandomCrop((imgW // self.down_sample_scale, imgH // self.down_sample_scale),
                                                    self.mask)

    def __call__(self, batch):
        images, label_strs = zip(*batch)

        images_hr = [self.transform(image) for image in images]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images]
        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)


        return images_hr, images_lr, label_strs



class alignCollate_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, images_HRy, images_lry, label_strs = zip(*batch)

        new_images_HR = []
        new_images_LR = []
        new_label_strs = []
        if type(images_HR[0]) == list:
            for image_item in images_HR:
                new_images_HR.extend(image_item)

            for image_item in images_lr:
                new_images_LR.extend(image_item)

            for image_item in label_strs:
                new_label_strs.extend(image_item)

            images_HR = new_images_HR
            images_lr = new_images_LR
            label_strs = new_label_strs

        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_HR, images_lr, label_strs


class alignCollate_realWTL(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, images_HRy, images_lry, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        images_lry = [self.transform2(image) for image in images_lry]
        images_lry = torch.cat([t.unsqueeze(0) for t in images_lry], 0)

        images_HRy = [self.transform(image) for image in images_HRy]
        images_HRy = torch.cat([t.unsqueeze(0) for t in images_HRy], 0)

        max_len = 26

        label_batches = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                padding = int(inter_com / (len(word) - 1))
                new_word = word[0]
                for i in range(len(word) - 1):
                   new_word += "-" * padding + word[i+1]

                word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            if labels.shape[0] > 0:
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
            else:
                label_batches.append(label_vecs)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        return images_HR, images_lr, images_HRy, images_lry, label_strs, label_rebatches


class alignCollate_realWTLAMask(alignCollate_syn):

    def get_mask(self, image):
        img_hr = np.transpose(image.data.numpy() * 255, (1, 2, 0))
        img_hr_gray = cv2.cvtColor(img_hr[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # print("img_hr_gray: ", np.unique(img_hr_gray), img_hr_gray.shape)
        kernel = np.ones((5, 5), np.uint8)
        hr_canny = cv2.Canny(img_hr_gray, 20, 150)
        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        hr_canny = cv2.GaussianBlur(hr_canny, (5, 5), 1)
        weighted_mask = 0.4 + (hr_canny / 255.0) * 0.6
        return torch.tensor(weighted_mask).float().unsqueeze(0)

    def __call__(self, batch):
        images_HR, images_lr, images_HRy, images_lry, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        # images_pseudoLR = [self.transform2(image) for image in images_HR]
        # images_pseudoLR = torch.cat([t.unsqueeze(0) for t in images_pseudoLR], 0)

        images_pseudoLR = None

        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [self.transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        images_lry = [self.transform2(image) for image in images_lry]
        images_lry = torch.cat([t.unsqueeze(0) for t in images_lry], 0)

        images_HRy = [self.transform(image) for image in images_HRy]
        images_HRy = torch.cat([t.unsqueeze(0) for t in images_HRy], 0)

        # print("images_lry:", images_lry.shape)

        # weighted_masks = [self.get_mask(image_HR) for image_HR in images_HR]
        # weighted_masks = torch.cat([t.unsqueeze(0) for t in weighted_masks], 0)

        # print("weighted_masks:", weighted_masks.shape, np.unique(weighted_masks))
        max_len = 26

        label_batches = []
        weighted_masks = []
        weighted_tics = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                # word = [ch for ch in word]
                # word[2] = "e"
                # word = "".join(word)
                pass
            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                padding = int(inter_com / (len(word) - 1))
                new_word = word[0]
                for i in range(len(word) - 1):
                    new_word += "-" * padding + word[i+1]

                word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            #########################################
            # random.shuffle(label_list)
            #########################################
            
            if len(label_list) <= 0:
                # blank label
                weighted_masks.append(0)
            else:
                weighted_masks.extend(label_list)

            # word_len = len(word)
            # if word_len > max_len:
            #     max_len = word_len
            # print("label_list:", word, label_list)
            labels = torch.tensor(label_list)[:, None].long()

            # print("labels:", labels)

            if labels.shape[0] > 0:
                label_vecs = torch.zeros((labels.shape[0], self.alsize))
                # print(label_vecs.scatter_(-1, labels, 1))
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
                weighted_tics.append(1)
            else:
                label_vecs = torch.zeros((1, self.alsize))
                # Assign a blank label
                label_vecs[0, 0] = 1.
                label_batches.append(label_vecs)
                weighted_tics.append(0)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        return images_HR, images_pseudoLR, images_lr, images_HRy, images_lry, label_strs, label_rebatches, torch.tensor(weighted_masks).long(), torch.tensor(weighted_tics)


import random
class alignCollate_realWTL_withcrop(alignCollate_syn_withcrop):
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW

        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        HR_list = []
        LR_list = []

        for i in range(len(images_HR)):

            shift_proportion = 0.4 * random.random()
            l_shift = random.random() * shift_proportion
            r_shift = shift_proportion - l_shift
            interval = [l_shift, 1 - r_shift]
            HR_list.append(self.transform(images_HR[i], interval))
            LR_list.append(self.transform2(images_lr[i], interval))

        images_HR = torch.cat([t.unsqueeze(0) for t in HR_list], 0)
        images_lr = torch.cat([t.unsqueeze(0) for t in LR_list], 0)

        # images_HR = [self.transform(image) for image in images_HR]
        # images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        # images_lr = [self.transform2(image) for image in images_lr]
        # images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        max_len = 0

        label_batches = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                padding = int(inter_com / (len(word) - 1))
                new_word = word[0]
                for i in range(len(word) - 1):
                    new_word += "-" * padding + word[i+1]

                word = new_word
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            # shifting:
            # if len(label_list) > 2:
                #     if label_list[-1] > 0 and label_list[-1] < self.alsize - 1:
            #     label_list[-1] = 0

            word_len = len(word)
            if word_len > max_len:
                max_len = word_len
            # print("label_list:", word, label_list)
            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            if labels.shape[0] > 0:
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
            else:
                label_batches.append(label_vecs)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)
        noise = (torch.rand(label_rebatches.shape) - 0.5) * 0.2

        label_rebatches += noise

        return images_HR, images_lr, label_strs, label_rebatches


class alignCollateW2V_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs, w2vs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        image_masks = []
        image_lrs = []

        for i in range(len(images_lr)):
            image_lr = transform2(images_lr[i], label_strs[i])
            image_lrs.append(image_lr)
            # image_masks.append(image_mask)

        # images_lr = [transform2(images_lr[i], label_strs[i])[0] for i in range(len(images_lr))]
        images_lr = torch.cat([t.unsqueeze(0) for t in image_lrs], 0)
        # image_masks = torch.cat([t.unsqueeze(0) for t in image_masks], 0)

        images_HR = [transform(images_HR[i], label_strs[i]) for i in range(len(images_HR))]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        # print("Align:", type(w2vs), len(w2vs))
        # w2v_tensors = torch.cat([torch.tensor(w2v)[None, ...] for w2v in w2vs], 0).float()
        # print("Align:", type(w2vs), len(w2vs), w2v_tensors.shape)
        w2v_tensors = None

        # print("image_HR:", images_HR.shape, images_lr.shape)

        return images_HR, images_lr, label_strs, w2v_tensors # , image_masks


class alignCollatec2f_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)

        # print("images_HR:", images_HR[0], images_lr[0])

        image_MX = []

        for i in range(len(images_HR)):
            HR_i = np.array(images_HR[i]).astype(np.float32)
            LR_i = np.array(images_lr[i]).astype(np.float32)

            image_MX.append(Image.fromarray(((HR_i + LR_i) / 2.0).astype(np.uint8)))

            # print("unique:", np.unique(HR_i))
            # print("unique:", np.unique(LR_i))

        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        images_MX = [transform2(image) for image in image_MX]
        images_MX = torch.cat([t.unsqueeze(0) for t in images_MX], 0)

        # print("Align:", type(w2vs), len(w2vs))
        # w2v_tensors = torch.cat([torch.tensor(w2v)[None, ...] for w2v in w2vs], 0).float()
        # print("Align:", type(w2vs), len(w2vs), w2v_tensors.shape)

        return images_HR, images_lr, label_strs, images_MX


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


if __name__ == '__main__':
    # embed(header='dataset.py')

    import random

    # coding=utf-8
    # import cv2
    # import numpy as np

    dataset_list = []

    '''
    root_path = "/data0_ssd2t/majianqi/TextZoom/train2/"

    data_annos = lmdbDataset_real(root_path)
    nsamples = data_annos.nSamples

    save_dir = "canny/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in range(300):
        img_hr, img_lr, img_hry, img_lry, label_str = data_annos[i]

        img_hr = np.array(img_hr)
        img_lr = np.array(img_lr)

        img_hr_gray = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
        img_lr_gray = cv2.cvtColor(img_lr, cv2.COLOR_BGR2GRAY)

        # img = cv2.GaussianBlur(img, (3, 3), 0)

        img_hr_gray = cv2.resize(img_hr_gray, (128, 32))
        img_lr_gray = cv2.resize(img_lr_gray, (128, 32))

        randx = random.randint(0, 127)

        img_hr_gray[:, randx:] = img_lr_gray[:, randx:]

        kernel = np.ones((9, 9), np.uint8)

        hr_canny = cv2.Canny(img_hr_gray, 0, 255)
        lr_canny = cv2.Canny(img_lr_gray, 0, 255)

        hr_canny = cv2.dilate(hr_canny, kernel, iterations=1)
        lr_canny = cv2.dilate(lr_canny, kernel, iterations=1)

        hr_canny = cv2.GaussianBlur(hr_canny, (15, 15), 1)
        lr_canny = cv2.GaussianBlur(lr_canny, (15, 15), 1)

        pub_w = max(hr_canny.shape[1], lr_canny.shape[1])
        pub_h = hr_canny.shape[0] + lr_canny.shape[0] + 15 + lr_canny.shape[0]

        pub_img = np.zeros((pub_h, pub_w)).astype(np.uint8)
        pub_img[:lr_canny.shape[0], :lr_canny.shape[1]] = lr_canny
        pub_img[lr_canny.shape[0] + 5:lr_canny.shape[0] + 5 + hr_canny.shape[0], :hr_canny.shape[1]] = hr_canny
        pub_img[lr_canny.shape[0] * 2 + 10:lr_canny.shape[0] * 2 + 10 + hr_canny.shape[0], :hr_canny.shape[1]] = img_hr_gray
        print("kernel:", kernel.shape, np.unique(kernel), np.unique(pub_img))

        # cv2.imwrite(os.path.join(save_dir, 'Canny' + str(i) + '.jpg'), pub_img)

        cv2.imshow('pub_img', img_hr_gray)
        cv2.waitKey(0)
        # cv2.imshow('img_hr_gray', img_hr_gray)
        # cv2.imshow('Canny', canny)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    '''
    '''
    import json

    dataset_dir = "/data0_ssd2t/majianqi/RealSR_Final/"
    json_file = os.path.join(dataset_dir, "real_sr.json")

    with open(json_file, "r") as f:
        real_sr_annos = json.load(f)

    cnt = 0
    for anno_num in real_sr_annos:
        anno_obj = real_sr_annos[anno_num]

        # print("anno_obj:", anno_obj.keys())
        width = anno_obj["width"]
        height = anno_obj["height"]
        filename = anno_obj['rawFilename']

        split = "Test" if anno_obj["rawFilePath"] == "test" else "Train"
        rotate = anno_obj["rotate"]
        camera_type = filename.split("_")[0]
        image_path = os.path.join(dataset_dir, camera_type, split, "3", filename)
        print("image_path:", image_path)
        img = cv2.imread(image_path)

        if "polygons" in anno_obj:
            polygons = anno_obj["polygons"]

            word_rects = polygons["wordRect"]
            # print("polygons:", len(word_rects))
            cnt += len(word_rects)

            for wr in word_rects:
                print("wr:", wr)
                pts = [[float(pts["x"]), float(pts["y"])] for pts in wr["position"]]
                pts = np.array(pts).astype(np.int32)

                cv2.line(img, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), 255, 1)
                cv2.line(img, (pts[1, 0], pts[1, 1]), (pts[2, 0], pts[2, 1]), 255, 1)
                cv2.line(img, (pts[2, 0], pts[2, 1]), (pts[3, 0], pts[3, 1]), 255, 1)
                cv2.line(img, (pts[3, 0], pts[3, 1]), (pts[0, 0], pts[0, 1]), 255, 1)

        # print("img:", img)
        cv2.imshow("img:", img)
        cv2.waitKey(0)
    print("All instances:", cnt)
    '''
    pass