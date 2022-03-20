from .model_transformer import *

# coding=utf-8
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math, copy
import numpy as np
import time
from torch.autograd import Variable
import torchvision.models as models

from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead


torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)

n_class = 0
from .transformer_v2 import Transformer as Transformer_V2
from .transformer_v2 import TPTransformer, InfoTransformer

# from .tsrn import RecurrentResidualBlock
# from .tsrn import TSRN_TL_Encoder, TSRNEncoder

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False).to(x.device)
        return self.dropout(x)


class PositionalEncoding_learn(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding_learn, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        # self.register_parameter('pe', pe)

    def forward(self, x):
        # print(x.shape)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # cnt = 0
        # for l , x in zip(self.linears, (query, key, value)):
        #     print(cnt,l,x)
        #     cnt += 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # print("在Multi中，query的尺寸为", query.shape)
        # print("在Multi中，key的尺寸为", key.shape)
        # print("在Multi中，value的尺寸为", value.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    '''
    这里使用偏移k=2是因为前面补位embedding
    '''
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    # print(mask)
    # print("在attention模块,q_{0}".format(query.shape))
    # print("在attention模块,k_{0}".format(key.shape))
    # print("在attention模块,v_{0}".format(key.shape))
    # print("mask :",mask)
    # print("mask的尺寸为",mask.shape)

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # print("scores ", scores)
        '''
        可视化
        '''
        # print_scores = scores[0]
        # print(print_scores)
    # print("scoreshape",scores.shape)

    p_attn = F.softmax(scores, dim=-1)

    # if mask is not None:
    #     print("p_attn",p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # print("value:", p_attn.shape, value.shape)

    return torch.matmul(p_attn, value), p_attn


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return F.softmax(self.proj(x))
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        # print("embed",embed)
        # embed = self.lut(x)
        # print(embed.requires_grad)
        return embed


class Decoder(nn.Module):

    def __init__(self, feature_size, head_num=16, dropout=0.1):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm1 = LayerNorm(features=feature_size)

        self.multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm2 = LayerNorm(features=feature_size)

        self.pff = PositionwiseFeedForward(feature_size, feature_size)
        self.mul_layernorm3 = LayerNorm(features=feature_size)

        self.feature_size = feature_size

    def forward(self, text, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''

        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text

        origin_result = result
        result = result
        # print("global_info:", text.shape, conv_feature.shape, self.feature_size)
        result = self.mul_layernorm1(origin_result + self.mask_multihead(result, result, result, mask=mask))

        # b, c, h, w = conv_feature.shape
        # conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        origin_result = result
        result = result

        result = self.mul_layernorm2(origin_result + self.multihead(result, conv_feature, conv_feature, mask=None))

        origin_result = result
        result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))

        return result


class CatFetDecoder(nn.Module):

    def __init__(self, feature_size, head_num=16, dropout=0.1):
        super(CatFetDecoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm1 = LayerNorm(features=feature_size)

        self.multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm2 = LayerNorm(features=feature_size)

        self.pff = PositionwiseFeedForward(feature_size, feature_size)
        self.mul_layernorm3 = LayerNorm(features=feature_size)

    def forward(self, text, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''

        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text

        origin_result = result
        result = result

        print("decoder1:", origin_result.shape, result.shape, mask.shape)

        result = self.mul_layernorm1(origin_result + self.mask_multihead(result, result, result, mask=mask))

        b, c, h, w = conv_feature.shape
        # print("global_info:", global_info.shape, result.shape, conv_feature.shape)
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        origin_result = result
        result = result

        # print("origin_result:", origin_result.shape, result.shape, conv_feature.shape)

        result = self.mul_layernorm2(origin_result + self.multihead(result, conv_feature, conv_feature, mask=None))

        origin_result = result
        result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))

        return result


class Encoder(nn.Module):

    def __init__(self, output_channel=512, input_channel=256, global_pooling_size=(2, 35), encoder2D=None):
        super(Encoder, self).__init__()

        # self.avgpool = nn.AvgPool2d(global_pooling_size, global_pooling_size)

        self.cnn_bottleneck = nn.Conv2d(input_channel, output_channel, 1)
        self.bn_bottleneck = nn.BatchNorm2d(output_channel)
        self.relu_bottleneck = nn.ReLU()

        self.encoder2D = encoder2D
        self.pe_2D = positionalencoding2d(output_channel, global_pooling_size[0], global_pooling_size[1])

    def forward(self, feature):
        conv_result = feature
        b, c, h, w = conv_result.shape

        result = conv_result
        # result = self.avgpool(result).squeeze(2).squeeze(2)

        result = result.view(b, c, h * w)
        result = torch.mean(result, 2)

        result = result.unsqueeze(2).unsqueeze(2).contiguous()

        conv_result = self.relu_bottleneck(self.bn_bottleneck(self.cnn_bottleneck(conv_result)))
        pe_2D = self.pe_2D.to(conv_result.device)
        if not self.encoder2D is None:
            # (batch, 512, H, W)

            conv_result = conv_result + pe_2D  # [:, :h, :w]
            conv_result = conv_result.view(b, c, -1)
            conv_feature_enhanced = self.encoder2D(conv_result)
            conv_result = conv_feature_enhanced.view(b, c, h, w)

        return conv_result, result


class FeatureEnhancer(nn.Module):

    def __init__(self, feature_size, head_num, dropout):
        super(FeatureEnhancer, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm1 = LayerNorm(features=feature_size)

        self.pff = PositionwiseFeedForward(feature_size, feature_size)
        self.mul_layernorm3 = LayerNorm(features=feature_size)

    def forward(self, conv_feature):
        '''
        conv_feature: (batch, channel, H * W)
        '''
        # (N, C, T) -> (N, T, C)
        result = conv_feature.permute(0, 2, 1).contiguous()

        origin_result = result
        result = result
        result = self.mul_layernorm1(origin_result + self.mask_multihead(result, result, result, mask=None))

        origin_result = result
        result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))

        # (N, T, C) -> (N, C, T)
        return result.permute(0, 2, 1).contiguous()


class FeatureEnhancerW2V(nn.Module):

    def __init__(self, vec_d, feature_size, head_num, dropout):
        super(FeatureEnhancerW2V, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm1 = LayerNorm(features=feature_size)

        self.pff = PositionwiseFeedForward(feature_size, feature_size)
        self.mul_layernorm3 = LayerNorm(features=feature_size)

        self.w2v_proj = nn.Linear(vec_d, feature_size)

    def forward(self, conv_feature, word2vec):
        '''
        conv_feature: (batch, channel, H * W)
        '''
        # (N, C, T) -> (N, T, C)
        result = conv_feature.permute(0, 2, 1).contiguous()

        # print("result:", result.shape)

        # vx = self.w2v_proj(word2vec)

        # print("result:", result.shape, vx.shape)

        # result += vx[:, None]

        origin_result = result
        result = result
        result = self.mul_layernorm1(origin_result + self.mask_multihead(result, result, result, mask=None))

        origin_result = result
        result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))

        # (N, T, C) -> (N, C, T)
        return result.permute(0, 2, 1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels, res_channel, downsample=False):
        super(ResidualBlock, self).__init__()

        self.downsample = downsample

        if downsample:
            self.down_conv = nn.Conv2d(channels, res_channel, kernel_size=3, padding=1, stride=2)

        self.conv1 = nn.Conv2d(res_channel, res_channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(res_channel)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(res_channel, res_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(res_channel)

    def forward(self, x):

        if self.downsample:
            x = self.down_conv(x)

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # print("Block X:", x.shape, residual.shape)

        return x + residual


class ResidualTransBlock(nn.Module):
    def __init__(self, channels, res_channel, d_model=1024, downsample=False):
        super(ResidualTransBlock, self).__init__()

        self.downsample = downsample

        self.transformer = Transformer_V2(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=1024,
                                          num_encoder_layers=0,
                                          num_decoder_layers=1,
                                          normalize_before=False,
                                          return_intermediate_dec=True, )

        if downsample:
            self.down_conv = nn.Conv2d(channels, res_channel, kernel_size=3, padding=1, stride=2)

        self.conv1 = nn.Conv2d(res_channel, res_channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(res_channel)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(res_channel, res_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(res_channel)

    def forward(self, x, mask, query_emb, pe, tgt=None):

        if self.downsample:
            x = self.down_conv(x)

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)

        N, C, H, W = residual.shape

        residual = residual.view(N, C * H, W).permute(2, 0, 1)
        # src #
        if not tgt is None:
            tgt = tgt.view(N, C * H, W).permute(2, 0, 1)
        hs = self.transformer(residual, mask, query_emb, pe, tgt=tgt).mean(0)
        # [W, N, C * H] -> [N, C, H, W]
        residual = hs.permute(1, 2, 0).view(N, C, H, W)

        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


def _get_acctivation(name):
    if name == 'relu':
        return F.relu
    elif name == 'sigmoid':
        return F.sigmoid


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, in_channel=4, out_channel=64):
        super(ConvEncoder, self).__init__()

        self.start_conv = nn.Conv2d(in_channel, 64, kernel_size=9, padding=4)
        self.relu = nn.PReLU()
        self.block1 = ResidualBlock(64, 64, downsample=False)
        self.block2 = ResidualBlock(64, out_channel, downsample=False)

    def forward(self, x):
        x = self.start_conv(x)
        x = self.relu(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)

        return [x1, x2]


class SRResConvEncoder(nn.Module):
    def __init__(self, in_channel=4, out_channel=64, upsample_block_num=1):
        super(SRResConvEncoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=9, padding=4),
            nn.PReLU()
        )
        # self.block2 = ResidualBlock(out_channel, out_channel)
        # self.block3 = ResidualBlock(out_channel, out_channel)
        # self.block4 = ResidualBlock(out_channel, out_channel)
        # self.block5 = ResidualBlock(out_channel, out_channel)
        # self.block6 = ResidualBlock(out_channel, out_channel)
        self.block7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        block1 = self.block1(x)
        # block2 = self.block2(block1)
        # block3 = self.block3(block2)
        # block4 = self.block4(block3)
        # block5 = self.block5(block4)
        # block6 = self.block6(block5)
        #block7 = self.block7(block6)

        return [block1] # , block3, block4, block5, block6, block1 + block7


class SRResTransConvEncoder(nn.Module):
    def __init__(self, in_channel=4, out_channel=64, d_model=1024):
        super(SRResTransConvEncoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualTransBlock(out_channel, out_channel, d_model=d_model)
        self.block3 = ResidualTransBlock(out_channel, out_channel, d_model=d_model)
        self.block4 = ResidualTransBlock(out_channel, out_channel, d_model=d_model)
        self.block5 = ResidualTransBlock(out_channel, out_channel, d_model=d_model)
        self.block6 = ResidualTransBlock(out_channel, out_channel, d_model=d_model)
        self.block7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x, mask, query_emb, pe, tgt=None):
        # block1 = self.block1(x)
        block2 = self.block2(x, mask, query_emb, pe, tgt)
        block3 = self.block3(block2, mask, query_emb, pe, block2)
        block4 = self.block4(block3, mask, query_emb, pe, block3)
        block5 = self.block5(block4, mask, query_emb, pe, block4)
        block6 = self.block6(block5, mask, query_emb, pe, block5)
        block7 = self.block7(block6)

        return [block2, block3, block4, block5, block6, block7]


class SRResConvDecoder(nn.Module):

    def __init__(self, in_channel=64, out_channel=4, upsample_block_num=1):
        super(SRResConvDecoder, self).__init__()

        block8 = [UpsampleBLock(in_channel, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(in_channel, out_channel, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = self.block8(x)
        return F.tanh(x)


class ConvDecoder(nn.Module):
    def __init__(self, in_channel=64, out_channel=4):
        super(ConvDecoder, self).__init__()

        self.tconv1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        # nn.ConvTranspose2d(in_channel, 64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.tconv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.relu = nn.PReLU()

        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=9, padding=4)
        # nn.ConvTranspose2d(64, out_channel, 4, 2, 1, bias=False)
        # nn.Conv2d(16, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        trans_x1 = x

        x1 = self.tconv1(trans_x1)  #
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        # x2 = self.tconv2(x1 + trans_x1)
        # x2 = self.bn2(x2)
        # x2 = self.relu(x2)

        x2 = self.final_conv(x1)

        return F.tanh(x2)


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

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, (2, 1), padding=1, bias=False)
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


class InfoGenTrans(nn.Module):
    def __init__(
                self,
                t_emb,
                out_text_channels,
                output_size=(16, 64),
                # d_model=512,
                t_encoder_num=1,
                t_decoder_num=2,
                 ):
        super(InfoGenTrans, self).__init__()

        d_model = out_text_channels # * output_size[0]

        self.fc_in = nn.Linear(t_emb, d_model)
        self.activation = nn.PReLU()

        self.upsample_transformer = InfoTransformer(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=64,
                                          num_encoder_layers=t_encoder_num,
                                          num_decoder_layers=t_decoder_num,
                                          normalize_before=False,
                                          return_intermediate_dec=True, )

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0]
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.dropout = nn.Dropout(0.3)

        self.masking = torch.ones(output_size)

        # self.tp_uper = InfoGen(t_emb, out_text_channels)

    def forward(self, image_feature, tp_input):

        H, W = self.output_size
        x = tp_input

        # x_tar = tp_input
        # x_tar = self.tp_uper(x_tar)
        N_i, C_i, H_i, W_i = image_feature.shape

        x_tar = image_feature #F.interpolate(x_tar, (H_i, W_i), mode='bilinear', align_corners=True)
        # x_im = None
        # x_im = x_tar.view(N_i, C_i * H_i, W_i).permute(2, 0, 1)
        # [1024, N, 64]
        x_im = x_tar.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)
        # print("masking1:", self.masking.shape, self.masking)

        # if self.training:
        #     # x_im: [N, C, H, W]
        #     masking = self.dropout(self.masking).unsqueeze(0).unsqueeze(0)
        # else:
        #     masking = self.masking

        # print("masking:", image_feature.shape)

        device = x.device
        # print('x:', x.shape)
        x = x.permute(0, 3, 1, 2).squeeze(-1)
        x = self.activation(self.fc_in(x))
        N, L, C = x.shape

        x_pos = self.pe(torch.zeros((N, L, C)).to(device)).permute(1, 0, 2)
        mask = torch.zeros((N, L)).to(device).bool()
        x = x.permute(1, 0, 2)

        # print("x_im:", x_im.shape)

        text_prior, pr_weights = self.upsample_transformer(x, mask, self.init_factor.weight, x_pos, tgt=x_im)
        text_prior = text_prior.mean(0)

        # [N, L, C] -> [N, C, H, W]

        text_prior = text_prior.permute(1, 2, 0).view(N, C, H, W) #// H

        return text_prior, pr_weights



class SRTransformer_V4(nn.Module):

    def __init__(self,
                 in_planes=4,
                 d_model=1024,
                 cnt_d_model=64,
                 seq_len=64,
                 t_encoder_num=4,
                 t_decoder_num=8,
                 STN=False,
                 scale_factor=2,
                 t_emb=37
                 ):
        super(SRTransformer_V4, self).__init__()

        self.conv_encoder = ConvEncoder()  # SRResConvEncoder() #TSRNEncoder() #S
        self.conv_decoder = SRResConvDecoder()  # in_channel=32, upsample_block_num=0

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)
        self.tp_pe = PositionalEncoding(d_model=cnt_d_model, dropout=0.1, max_len=5000)

        self.t_encoder_num = t_encoder_num
        self.t_decoder_num = t_decoder_num

        self.seq_len = seq_len
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.transformer = Transformer_V2(d_model=d_model,
                                         dropout=0.1,
                                         nhead=4,
                                         dim_feedforward=1024,
                                         num_encoder_layers=t_encoder_num,
                                         num_decoder_layers=t_decoder_num,
                                         normalize_before=False,
                                         return_intermediate_dec=True, )
        self.text_cls = t_emb
        # self.infoGen = InfoGen(t_emb, output_size=32)

        self.infoGen = InfoGenTrans(self.text_cls, out_text_channels=cnt_d_model)
        self.activation = nn.ReLU()

        self.out_size = [64, 16]

        self.tps_inputsize = [self.out_size[1], self.out_size[0]]

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1)

        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(self.tps_inputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_LR, text_prior=None):

        # LR_conv_feature

        # embed()
        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(img_LR)
            img_LR, _ = self.tps(img_LR, ctrl_points_x)

        LR_conv_feature = img_LR
        LR_conv_feature = self.conv_encoder(LR_conv_feature)[-1]
        '''
        if self.training:
            HR_conv_feature = torch.nn.functional.interpolate(img_HR, (img_HR.shape[-2] // 2, img_HR.shape[-1] // 2), mode='bilinear')
            HR_conv_feature = self.conv_encoder(HR_conv_feature)[-1]

            # [16, 64] -> [4, 16]
            LR_feat = self.pool4(LR_conv_feature)
            HR_feat = self.pool4(HR_conv_feature).detach()
        '''

        text_prior, pr_weights = self.infoGen(LR_conv_feature, text_prior)

        # LR_conv_feature = self.activation(LR_conv_feature + text_prior)

        device = LR_conv_feature.device

        src = self.activation(LR_conv_feature + text_prior)
        N, C, H, W = src.shape
        # [N, C, H, W] - > [H * W, N, C]
        src = src.view(N, C * H, W).permute(2, 0, 1)
        text_prior = text_prior.reshape(N, C * H, W).permute(2, 0, 1)
        LR_conv_feature = LR_conv_feature.reshape(N, C * H, W).permute(2, 0, 1)

        # src_tp = self.activation(self.infoGen(text_prior.permute(0, 3, 1, 2).squeeze(-1))).permute(1, 0, 2)
        # print("src_tp: ", src_tp.shape, src.shape)
        # spatial_t_emb = self.infoGen(text_prior)
        # src_tp = F.interpolate(spatial_t_emb, (H, W), mode='bilinear', align_corners=True)
        # print("src_tp: ", src_tp.shape)
        # src_tp = src_tp.view(N, C, H * W).permute(2, 0, 1)

        L, N, C_ = src.shape
        # L_T, N_T, C_T = src_tp.shape

        src_pos = self.pe(torch.zeros((N, L, C_)).to(device)).permute(1, 0, 2)

        mask = torch.zeros((N, L)).to(device).bool()

        # src_tp: [N, C, H, W]
        hs = self.transformer(text_prior, mask, self.init_factor.weight, src_pos, tgt=LR_conv_feature)  # + src_pos
        hs, attn_w = hs
        hs = hs.mean(0)
        # print("hs:", hs.shape, attn_w.shape)
        # [W, N, C * H] -> [N, C, H, W]
        hs = hs.permute(1, 2, 0).view(N, C, H, W)
        src = src.permute(1, 2, 0).view(N, C, H, W)
        # hs = hs.permute(1, 2, 0).view(N, int(C // 2), H * 2, W * 2)

        img_SR = self.conv_decoder(hs + src)

        # if self.training:
        #     return img_SR
        # else:
        #    return img_SR
        return img_SR, pr_weights


class SRTransformer_V3(nn.Module):

    def __init__(self,
                 in_planes=4,
                 d_model=1024,
                 seq_len=64,
                 t_encoder_num=3,
                 t_decoder_num=10,
                 STN=False,
                 scale_factor=2
                 ):
        super(SRTransformer_V3, self).__init__()

        self.conv_encoder = TSRNEncoder()  # TSRNEncoder() #SRResConvEncoder
        self.conv_decoder = SRResConvDecoder()  # in_channel=32, upsample_block_num=0

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)
        self.t_encoder_num = t_encoder_num
        self.t_decoder_num = t_decoder_num

        self.seq_len = seq_len
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.transformer = Transformer_V2(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=1024,
                                          num_encoder_layers=t_encoder_num,
                                          num_decoder_layers=t_decoder_num,
                                          normalize_before=False,
                                          return_intermediate_dec=True, )

        self.out_size = [64, 16]

        self.tps_inputsize = [self.out_size[1], self.out_size[0]]

        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(self.tps_inputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_LR, img_HR=None):

        # LR_conv_feature

        # embed()
        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(img_LR)
            img_LR, _ = self.tps(img_LR, ctrl_points_x)

        LR_conv_feature = img_LR
        LR_conv_feature = self.conv_encoder(LR_conv_feature)[-1]
        '''
        if self.training:
            HR_conv_feature = torch.nn.functional.interpolate(img_HR, (img_HR.shape[-2] // 2, img_HR.shape[-1] // 2), mode='bilinear')
            HR_conv_feature = self.conv_encoder(HR_conv_feature)[-1]

            # [16, 64] -> [4, 16]
            LR_feat = self.pool4(LR_conv_feature)
            HR_feat = self.pool4(HR_conv_feature).detach()
        '''
        device = LR_conv_feature.device

        src = LR_conv_feature
        N, C, H, W = src.shape
        # [N, C, H, W] - > [W, N, C * H]
        src = src.view(N, C * H, W).permute(2, 0, 1)

        mask = torch.zeros((N, W)).to(device).bool()
        # Dimension aligned from [N, L, C] - > [L, N, C]
        L, N, C_ = src.shape
        src_pos = self.pe(torch.zeros((N, L, C_)).to(device)).permute(1, 0, 2)
        hs = self.transformer(src, mask, self.init_factor.weight, src_pos).mean(0)  #

        # [W, N, C * H] -> [N, C, H, W]
        hs = hs.permute(1, 2, 0).view(N, C, H, W)
        # hs = hs.permute(1, 2, 0).view(N, int(C // 2), H * 2, W * 2)

        img_SR = self.conv_decoder(hs)

        if self.training:
            return img_SR, None
        else:
            return img_SR


class SRTransformer_V2(nn.Module):

    def __init__(self,
                 in_planes=4,
                 d_model=1024,
                 seq_len=64,
                 t_encoder_num=3,
                 t_decoder_num=10,
                 STN=False,
                 scale_factor=2
                 ):
        super(SRTransformer_V2, self).__init__()

        self.conv_encoder = SRResConvEncoder()  # TSRNEncoder() #S
        self.conv_decoder = SRResConvDecoder()  # in_channel=32, upsample_block_num=0

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)
        self.t_encoder_num = t_encoder_num
        self.t_decoder_num = t_decoder_num

        self.seq_len = seq_len
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.transformer = Transformer_V2(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=1024,
                                          num_encoder_layers=t_encoder_num,
                                          num_decoder_layers=t_decoder_num,
                                          normalize_before=False,
                                          return_intermediate_dec=True, )

        self.out_size = [64, 16]

        self.tps_inputsize = [self.out_size[1], self.out_size[0]]

        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(self.tps_inputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_LR, img_HR=None):

        # LR_conv_feature

        # embed()
        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(img_LR)
            img_LR, _ = self.tps(img_LR, ctrl_points_x)

        LR_conv_feature = img_LR
        # torch.nn.functional.interpolate(img_LR, (img_LR.shape[-2] * 2, img_LR.shape[-1] * 2), mode='bilinear')
        # print('LR_conv_feature:', LR_conv_feature.shape, img_LR.shape)
        LR_conv_feature = self.conv_encoder(LR_conv_feature)[-1]  # [-self.t_encoder_num:] #-1

        device = LR_conv_feature.device

        src = LR_conv_feature
        N, C, H, W = src.shape
        # [N, C, H, W] - > [W, N, C * H]
        src = src.view(N, C * H, W).permute(2, 0, 1)

        mask = torch.zeros((N, W)).to(device).bool()
        src_pos = self.pe(src)

        hs = self.transformer(src, mask, self.init_factor.weight, src_pos).mean(0)

        # [W, N, C * H] -> [N, C, H, W]
        hs = hs.permute(1, 2, 0).view(N, C, H, W)
        # hs = hs.permute(1, 2, 0).view(N, int(C // 2), H * 2, W * 2)

        img_SR = self.conv_decoder(hs)

        return img_SR


class SRTransformer(nn.Module):

    def __init__(self, feature_size=512, vec_d=300, t_encoder_num=2, t_decoder_num=2):
        super(SRTransformer, self).__init__()

        # self.embedding_radical = Embeddings(int(feature_size / 2), n_class)
        self.pe_256 = PositionalEncoding(d_model=1024, dropout=0.1, max_len=5000)
        self.pe_64 = PositionalEncoding(d_model=1024, dropout=0.1, max_len=5000)
        self.pe_16 = PositionalEncoding(d_model=16, dropout=0.1, max_len=5000)

        self.pe_list = [
            self.pe_64, self.pe_256
        ]

        self.conv_encoder = SRResConvEncoder()
        self.conv_decoder = SRResConvDecoder()

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.decoders = nn.ModuleList([
            Decoder(
                1024,
                head_num=4,
                dropout=True,
            ),
            Decoder(
                1024,
                head_num=4,
                dropout=True,
            )
        ])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.attribute = None

    def get_attribute_grad(self):
        print(self.attribute.grad_fn)

    def make_embeddings(self, input_features, right_shift=False):

        # Ensure input feature to have shape with [N, C, H, W]

        embeddings = []

        for i in range(len(input_features)):

            feat_in = input_features[i]

            N, C, H, W = feat_in.shape
            device = feat_in.device

            # print("Shape:", H, W)
            # [N, C, H, W] -> [N, H*W, C]
            ###
            # emb_seq = feat_in.permute(0, 2, 3, 1).view(N, H * W, C)
            ###

            # [N, C, H, W] -> [N, W, C * H]

            emb_seq = feat_in.view(N, C * H, W).permute(0, 2, 1)

            if right_shift:
                start_status = torch.zeros((N, 1, C * H)).to(device)
                emb_seq = torch.cat([start_status, emb_seq[:, :W - 1, :]], 1) * 0.

            # print("i", i, feat_in.shape, emb_seq.shape)

            emb_seq = self.pe_list[i](emb_seq)
            embeddings.append(emb_seq)

        return embeddings

    def forward(self, img_LR, img_HR=None):

        # LR_conv_feature

        LR_conv_feature = img_LR
        # torch.nn.functional.interpolate(img_LR, (img_LR.shape[-2] * 2, img_LR.shape[-1] * 2), mode='bilinear')
        # print('LR_conv_feature:', LR_conv_feature.shape, img_LR.shape)
        LR_conv_features = self.conv_encoder(LR_conv_feature)

        # pixel shuffle first
        # LR_conv_features_shuffled = [self.pixel_shuffle(feat) for feat in LR_conv_features]
        LR_embeddings = self.make_embeddings(LR_conv_features)

        results = []

        # in: [1, 1/2, 1/4]
        # out: [2, 1, 2/1]

        if True:  # self.training
            # HR_conv_feature
            HR_conv_feature = img_HR
            HR_conv_features = self.conv_encoder(HR_conv_feature)
            HR_embeddings = self.make_embeddings(HR_conv_features, right_shift=True)

            for i in range(len(LR_embeddings)):
                N, C, H, W = LR_conv_features[i].shape

                HR_embedding = HR_embeddings[i]
                LR_embedding = LR_embeddings[i]

                result = self.decoders[i](HR_embedding, LR_embedding)
                # print(i, "result:", result.shape, HR_embedding.shape, LR_embedding.shape, LR_conv_features[i].shape)
                # result = result.view(N, H, W, C).permute(0, 3, 1, 2)
                # [N, W, C * H] - > [N, C, H, W]
                result = result.permute(0, 2, 1).view(N, C, H, W)
                results.append(result)

        else:
            for i in range(len(LR_conv_features)):
                feat_in = LR_conv_features[i]
                N, C, H, W = feat_in.shape
                device = feat_in.device

                LR_embedding = LR_embeddings[i]

                result_decode = result = torch.zeros((N, 1, C)).to(device)

                for n in range(H * W):
                    result_decode = self.decoders[i](self.pe_list[i](result), LR_embedding)
                    result = torch.cat([result, result_decode[:, -1:, :]], 1)
                    # print("inference:", str(n) + " / " + str(H * W), result.shape)

                result_decode = result_decode.view(N, H, W, C).permute(0, 3, 1, 2)
                results.append(result_decode)

        results.append(LR_conv_features[-1])

        img_SR = self.conv_decoder(results)

        return img_SR

