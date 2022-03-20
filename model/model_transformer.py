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

torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)

n_class = 0


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
                         requires_grad=False)
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

    p_attn = F.softmax(scores, dim = -1)

    # if mask is not None:
    #     print("p_attn",p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
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
        super(Decoder,self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm1 = LayerNorm(features=feature_size)

        self.multihead = MultiHeadedAttention(h=head_num, d_model=feature_size, dropout=dropout)
        self.mul_layernorm2 = LayerNorm(features=feature_size)

        self.pff = PositionwiseFeedForward(feature_size, feature_size)
        self.mul_layernorm3 = LayerNorm(features=feature_size)

    def forward(self, text, global_info, conv_feature, text_length):
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
        # print("global_info:", global_info.shape, result.shape)
        result = self.mul_layernorm1(origin_result + self.mask_multihead(result, result, result, mask=mask))

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        origin_result = result
        result = result

        # print("origin_result:", origin_result.shape, result.shape, conv_feature.shape)

        result = self.mul_layernorm2(origin_result + self.multihead(result, conv_feature, conv_feature, mask=None))

        origin_result = result
        result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))
        # result = origin_result + result

        # origin_result = result
        # # result =
        # result = self.mul_layernorm1(origin_result + self.mask_multihead(result, result, result, mask=mask))
        #
        # b, c, h, w = conv_feature.shape
        # conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        # origin_result = result
        # result = self.mul_layernorm2(origin_result + self.multihead(result, conv_feature, conv_feature, mask=None))
        #
        # origin_result = result
        # result = self.mul_layernorm3(origin_result + self.pff(result))
        # # result = origin_result + result

        return result


class CatFetDecoder(nn.Module):

    def __init__(self, feature_size, head_num=16, dropout=0.1):
        super(CatFetDecoder,self).__init__()

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

            conv_result = conv_result + pe_2D #[:, :h, :w]
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


class Transformer(nn.Module):

    def __init__(self, cfg, n_class, feature_size=512):
        super(Transformer, self).__init__()

        self.embedding_radical = Embeddings(int(feature_size / 2), n_class)

        self.pe = PositionalEncoding(d_model=int(feature_size / 2), dropout=0.1, max_len=5000)

        feature_reso = cfg.MODEL.ROI_REC_HEAD.POOLER_RESOLUTION

        self.feature_2Datt = None
        if cfg.MODEL.ROI_REC_HEAD.TRANSFORMER.FEATURE_2DATT:
            self.feature_2Datt = FeatureEnhancer(
                feature_size=feature_size,
                head_num=cfg.MODEL.ROI_REC_HEAD.TRANSFORMER.HEAD_NUM,
                dropout=cfg.MODEL.ROI_REC_HEAD.TRANSFORMER.DROPOUT
            )

        self.encoder = Encoder(
            output_channel=feature_size,
            global_pooling_size=(int(feature_reso[0] / 2), feature_reso[1]),
            encoder2D=self.feature_2Datt
        )

        decoder = CatFetDecoder(
            feature_size,
            head_num=cfg.MODEL.ROI_REC_HEAD.TRANSFORMER.HEAD_NUM,
            dropout=cfg.MODEL.ROI_REC_HEAD.TRANSFORMER.DROPOUT,
        )
        self.decoders = clones(decoder, 1)

        self.generator_radical = Generator(feature_size, n_class)

        # Official init from torch repo
        # print("Initializing weights...")
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.attribute = None

    def get_attribute_grad(self):
        # print("attribute的值为",self.attribute)
        print(self.attribute.grad_fn)

    def forward(self, feature, text_length, text_input):

        conv_feature, global_info = self.encoder(feature)

        # print("conv_feature:", conv_feature.shape, global_info.shape)

        text = self.embedding_radical(text_input)
        blank = self.pe(torch.zeros(text.shape).cuda()).cuda()

        global_info = (global_info.squeeze(2).squeeze(2))[:, None].repeat(1, text.size(1), 1)
        # print("text:", text.shape, blank.shape, global_info.shape)
        result = torch.cat([text + blank, global_info], 2)
        batch, seq_len, _ = result.shape

        for decoder in self.decoders:
            result = decoder(result, global_info, conv_feature, text_length)
        result = self.generator_radical(result)

        return result


class ReasoningTransformer(nn.Module):

    def __init__(self, feature_size=512, vec_d=300):
        super(ReasoningTransformer, self).__init__()

        # self.embedding_radical = Embeddings(int(feature_size / 2), n_class)
        self.pe = PositionalEncoding(d_model=vec_d, dropout=0.1, max_len=5000)

        # feature_reso = cfg.MODEL.ROI_REC_HEAD.POOLER_RESOLUTION

        # self.feature_2Datt = None
        #v if True:
        self.feature_2Datt = FeatureEnhancer(
            feature_size=feature_size,
            head_num=4,
            dropout=True
        )

        self.encoder = Encoder(
            output_channel=feature_size,
            input_channel=feature_size,
            global_pooling_size=(16, 64),
            encoder2D=self.feature_2Datt
        )

        decoder = CatFetDecoder(
            feature_size,
            head_num=4,
            dropout=True,
        )
        self.decoders = clones(decoder, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.attribute = None

    def get_attribute_grad(self):
        print(self.attribute.grad_fn)

    def forward(self, feature, word_vector):

        conv_feature, global_info = self.encoder(feature)

        # print("conv_feature:", conv_feature.shape, global_info.shape)

        # word_vector -> text: [N, C] -> [N, 1, C] -> [N, H * W, C]
        total_stamp = conv_feature.size(2) * conv_feature.size(3)
        text = word_vector[:, None].repeat(1, total_stamp, 1)

        print("text:", text.shape)

        blank = self.pe(torch.zeros(text.shape).cuda()).cuda()

        # global_info: [N, C, 1, 1] - > [N, C] -> [N, T, C]
        global_info = (global_info.squeeze(2).squeeze(2))[:, None].repeat(1, text.size(1), 1)
        # print("text:", text.shape, blank.shape, global_info.shape)
        result = torch.cat([text + blank, global_info], 2)
        batch, seq_len, _ = result.shape

        for decoder in self.decoders:
            result = decoder(result, conv_feature)

        return result



def _get_acctivation(name):

    if name == 'relu':
        return F.relu
    elif name == 'sigmoid':
        return F.sigmoid


def test():

    transformer = Transformer().cuda()

    image = torch.Tensor(2, 3, 128, 400)
    image = image.cuda()
    text = torch.Tensor(2, 36).long()
    text = text.cuda()


    result = transformer(text, image)['result']
    print(result.shape)


def test_case_decoder():

    # mask = subsequent_mask(37)
    # print(mask)
    # exit(0)

    encoder = TransformOCR()
    image = torch.Tensor(2, 3, 128, 400)

    encoder = encoder.cuda()
    image = image.cuda()

    conv_feature, holistic_feature = encoder(image)

    text = torch.Tensor(2,36,512)
    text = text.cuda()
    # global_info = torch.Tensor(2,1,512).cuda()

    decoder = Decoder().cuda()

    result = decoder(text,holistic_feature,conv_feature)
    print(result.shape)





def test_case_encoder():

    encoder = TransformOCR()
    image = torch.Tensor(2,3,128,400)

    encoder = encoder.cuda()
    image = image.cuda()

    result = encoder(image)
    print(result.shape)


if __name__ == '__main__':
    test()
    pass
