import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR


def main(config, args, opt_TPG):
    Mission = TextSR(config, args, opt_TPG)

    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tsrn_tl_wmask', choices=['tsrn_tl_cascade_sft', 'tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'scgan', 'rdn', 'tbsrn',
                                                           'edsr', 'lapsrn', 'tsrn_tl_wmask', 'tsrn_tl_cascade', 'srcnn_tl', 'srresnet_tl', 'rdn_tl', 'vdsr_tl', 'tranSR_v4',
                                                                    "esrgan_tl", "scgan_tl", 'tbsrn_tl', 'tatt', "han", 'pcan', 'pcan_tl'])
    parser.add_argument('--go_test', action='store_true', default=False)
    parser.add_argument('--y_domain', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='../hard_space1/mjq/TextZoom/test/medium/', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--ic15sr', action='store_true', default=False, help='use IC15SR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--stu_iter', type=int, default=1, help='Default is set to 1, must be used with --arch=tsrn_tl_cascade')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--test_model', type=str, default='CRNN', choices=['ASTER', "CRNN", "MORAN"])
    parser.add_argument('--sr_share', action='store_true', default=False)
    parser.add_argument('--tpg_share', action='store_true', default=False)
    parser.add_argument('--use_label', action='store_true', default=False)
    parser.add_argument('--use_distill', action='store_true', default=False)
    parser.add_argument('--ssim_loss', action='store_true', default=False)
    parser.add_argument('--tpg', type=str, default="CRNN", choices=['CRNN', 'OPT'])
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    parser.add_argument('--CHNSR', action='store_true', default=False)
    parser.add_argument('--text_focus', action='store_true', default=False)
    parser.add_argument('--prob_insert', type=float, default=1., help='')
    parser.add_argument('--rotate_train', type=float, default=0., help='')
    parser.add_argument('--rotate_test', type=float, default=0., help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--badset', action='store_true', default=False)
    parser.add_argument('--training_stablize', action='store_true', default=False)
    parser.add_argument('--test_distorted_fusing', type=int, default=0)
    parser.add_argument('--results_rotate', action='store_true', default=False)
    parser.add_argument('--results_rotate_angle', type=float, default=5., help='')
    parser.add_argument('--learning_STN', action='store_true', default=False)
    parser.add_argument('--tssim_loss', action='store_true', default=False)
    parser.add_argument('--mse_fuse', action='store_true', default=False)
    parser.add_argument('--for_cascading', action='store_true', default=False)
    parser.add_argument('--color_loss', action='store_true', default=False)
    parser.add_argument('--BiSR', action='store_true', default=False)

    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    config.TRAIN.lr = args.learning_rate

    parser_TPG = argparse.ArgumentParser()
    #parser_TPG.add_argument('--exp_name', help='Where to store logs and models')
    #parser_TPG.add_argument('--train_data', required=True, help='path to training dataset')
    #parser_TPG.add_argument('--valid_data', required=True, help='path to validation dataset')
    #parser_TPG.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    #parser_TPG.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    #parser_TPG.add_argument('--batch_size', type=int, default=192, help='input batch size')
    #parser_TPG.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    #parser_TPG.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    #parser_TPG.add_argument('--saved_model', default='', help="path to model to continue training")
    #parser_TPG.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    #parser_TPG.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    #parser_TPG.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    #parser_TPG.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    #parser_TPG.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    #parser_TPG.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    #parser_TPG.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    #parser_TPG.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    #parser_TPG.add_argument('--select_data', type=str, default='MJ-ST',
    #                    help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    #parser_TPG.add_argument('--batch_ratio', type=str, default='0.5-0.5',
    #                    help='assign ratio for each selected data in the batch')
    #parser_TPG.add_argument('--total_data_usage_ratio', type=str, default='1.0',
    #                    help='total data usage ratio, this ratio is multiplied to total number of data.')
    #parser_TPG.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    #parser_TPG.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    #parser_TPG.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    #parser_TPG.add_argument('--rgb', action='store_true', help='use rgb input')
    #parser_TPG.add_argument('--character', type=str,
    #                    default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    #parser_TPG.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    #parser_TPG.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    #parser_TPG.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    #parser_TPG.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    #parser_TPG.add_argument('--FeatureExtraction', type=str, required=True,
    #                    help='FeatureExtraction stage. VGG|RCNN|ResNet')
    #parser_TPG.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    #parser_TPG.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    #parser_TPG.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    #parser_TPG.add_argument('--input_channel', type=int, default=1,
    #                    help='the number of input channel of Feature extractor')
    #parser_TPG.add_argument('--output_channel', type=int, default=512,
    #                    help='the number of output channel of Feature extractor')
    #parser_TPG.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # opt = parser_TPG.parse_args()

    opt = {
        "Transformation": 'None',
        "FeatureExtraction": 'ResNet',
        "SequenceModeling": 'None',
        "Prediction": 'CTC',
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
        "saved_model": "best_accuracy.pth",#"best_accuracy.pth", #"None-ResNet-None-CTC.pth",#"CRNN-PyTorchCTC.pth", # None-ResNet-None-CTC.pth
        "character": "-0123456789abcdefghijklmnopqrstuvwxyz"
    }

    if args.CHNSR:
        opt['character'] = open("al_chinese.txt", 'r').readlines()[0].replace("\n", "")
    opt["num_class"] = len(opt['character'])

    opt = EasyDict(opt)
    main(config, args, opt_TPG=opt)
