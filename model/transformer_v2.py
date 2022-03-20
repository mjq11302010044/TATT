# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable
import math
import numpy as np
# from .tsrn import mish
import cv2

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


class TPTransformer(nn.Module):

    def __init__(self, d_model=1024, cnt_d_model=64, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, feat_height=64, feat_width=64):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        tp_encoder_layer = TransformerEncoderLayer(cnt_d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        self.tp_encoder = TransformerEncoder(tp_encoder_layer, num_encoder_layers, encoder_norm)

        # ConvTransformerDecoderLayer
        decoder_layer = TransformerDualDecoderLayer(d_model, cnt_d_model, nhead, dim_feedforward,
                                                dropout, activation,
                                                normalize_before)  # ,feat_height=feat_height, feat_width=feat_width
        # sr_target_layer = RecurrentResidualBlockTL(d_model // feat_height, d_model // feat_height)

        # sr_target_layer = ResidualBlock(64, feat_height=feat_height, feat_width=feat_width)

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TPTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # src, src_tp, mask, tp_mask, self.init_factor.weight, src_pos, tp_pos,
    def forward(self, src, cnt_memory, mask, cnt_mask, query_embed, pos_embed, cnt_pos_embed, tgt=None, text_prior=None):
        # flatten NxCxHxW to HWxNxC
        w, bs, hc = src.shape
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        if tgt is None:
            tgt = torch.zeros_like(query_embed)

        # print("src_tp:", src_tp.shape, src.shape)

        cnt_memory = self.tp_encoder(cnt_memory, src_key_padding_mask=cnt_mask, pos=cnt_pos_embed)
        global_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # print("pos_embed:", pos_embed.shape, cnt_pos_embed.shape)

        hs = self.decoder(tgt, global_memory, cnt_memory, memory_key_padding_mask=mask, cnt_memory_key_padding_mask=cnt_mask,
                          pos=pos_embed, cnt_pos=cnt_pos_embed, query_pos=query_embed, text_prior=text_prior)  # src_tp

        return hs


class Transformer(nn.Module):

    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, feat_height=16, feat_width=64):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # ConvTransformerDecoderLayer
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation,
                                                normalize_before)#, feat_height=feat_height, feat_width=feat_width

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, tgt=None, text_prior=None):
        # flatten NxCxHxW to HWxNxC
        w, bs, hc = src.shape
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        if tgt is None:
            tgt = torch.zeros_like(query_embed)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs


class InfoTransformer(nn.Module):

    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, feat_height=16, feat_width=64):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # ConvTransformerDecoderLayer
        decoder_layer = TransformerDecoderLayer_TP(d_model, nhead, dim_feedforward,
                                                dropout, activation,
                                                normalize_before)#, feat_height=feat_height, feat_width=feat_width

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 1024
        self.gru_encoding = nn.GRU(d_model * feat_height, d_model * feat_height // 2, bidirectional=True, batch_first=True)
       

        # self.gru_encoding_horizontal = nn.GRU(d_model, d_model// 2, bidirectional=True,
        #                                                        batch_first=True)

        # self.gru_encoding_vertical = nn.GRU(d_model, d_model // 2, bidirectional=True,
        #                                       batch_first=True)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.feat_size = (feat_height, feat_width)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, tgt=None, text_prior=None, spatial_size=(16, 64)):
        # flatten NxCxHxW to HWxNxC
        w, bs, hc = src.shape
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # print("query_embed:", query_embed.shape)
        '''
        if not self.training:
            H, W = spatial_size
            up = int((W - H) / 2)
            bottom = H + int((W - H) / 2)
            query_embed = query_embed.reshape(self.feat_size[0], self.feat_size[1], bs, hc)
            query_embed = query_embed[up:bottom, ...]
            query_embed = query_embed.reshape(spatial_size[0] * spatial_size[1], bs, hc)
        '''

        # print("shape:", tgt.shape, query_embed.shape)

        query_embed = query_embed.reshape(self.feat_size[0], self.feat_size[1], bs, hc)\
            .permute(1, 2, 0, 3)\
            .reshape(self.feat_size[1], bs, self.feat_size[0] * hc)
        query_embed, _ = self.gru_encoding(query_embed)
        query_embed = query_embed.reshape(self.feat_size[1], bs, self.feat_size[0], hc)\
            .permute(2, 0, 1, 3)\
            .reshape(self.feat_size[0] * self.feat_size[1], bs, hc)

        '''
        query_embed = query_embed.reshape(self.feat_size[0], self.feat_size[1], bs, hc)
        #[H, B, C]
        query_embed_vertical = query_embed.mean(1)
        #[W, B, C]
        query_embed_horizontal = query_embed.mean(0)
        query_embed_vertical, _ = self.gru_encoding_vertical(query_embed_vertical)
        query_embed_horizontal, _ = self.gru_encoding_horizontal(query_embed_horizontal)
        # [H, 1, B, C] + [1, W, B, C]
        query_embed = query_embed_vertical.unsqueeze(1) + query_embed_horizontal.unsqueeze(0)
        query_embed = query_embed.reshape(self.feat_size[0] * self.feat_size[1], bs, hc)
        '''
        if tgt is None:
            tgt = torch.zeros_like(query_embed)

        # print("tgt:", tgt.shape)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        if type(src) == list:
            output = src[-1]
        else:
            output = src

        # print("q k", src.shape, src_key_padding_mask.shape)
        i = 0
        for layer in self.layers:
            if type(src) == list:
                src_item = src[i]
                i += 1
            else:
                src_item = src
            output = layer(output + src_item, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDualDecoder(nn.Module):

    def __init__(self, decoder_layer, sr_target_decoder, num_layers, norm=None, return_intermediate=False, height=16, width=64):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.sr_layers = _get_clones(sr_target_decoder, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.height = height
        self.width = width

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                ):
        output = tgt
        # sr_output = sr_src

        intermediate = []
        # sr_intermediate = []

        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            sr_layer = self.sr_layers[idx]
            # if not text_prior is None:
            #     output = output + self.norm(text_prior)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            L, N, C = output.shape

            # output_spa = output.permute(1, 2, 0).view(N, C // self.height, self.height, self.width)
            sr_output = sr_layer(output)
            output = self.norm(output + sr_output)
            # output_spa = output.permute(1, 2, 0).view(N, C // self.height, self.height, self.width)
            # output = output_spa.view(N, C * self.height, self.width).permute(2, 0, 1)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                # sr_intermediate.append(sr_output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, sr_output

_DEBUG = False

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                text_prior: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        attn_weights = None

        for layer in self.layers:

            # if not text_prior is None:
            #     output = output + self.norm(text_prior)

            output, attn_weights = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attn_weights

        return output, attn_weights

class TPTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, cnt_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                cnt_memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                cnt_memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                cnt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                text_prior: Optional[Tensor] = None):

        output = tgt

        intermediate = []
        attn_w = None

        for layer in self.layers:

            # if not text_prior is None:
            #     output = output + self.norm(text_prior)

            # print("pos:", pos.shape, cnt_pos.shape)

            output, attn_w = layer(output, memory, cnt_memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           cnt_memory_mask=cnt_memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           cnt_memory_key_padding_mask=cnt_memory_key_padding_mask,
                           pos=pos, cnt_pos=cnt_pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attn_w

        return output, attn_w

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # print("attn_mask:", src_key_padding_mask.shape)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDualDecoderLayer(nn.Module):

    def __init__(self, d_model, cnt_d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, spatial_size=(16, 64)):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cnt_multihead_attn = nn.MultiheadAttention(cnt_d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.fusion_linear = nn.Linear(d_model * 2, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.bg_norm = nn.LayerNorm(d_model)
        self.cnt_norm = nn.LayerNorm(d_model)

        self.bg_norm_after = nn.LayerNorm(d_model)
        self.cnt_norm_after = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.cnt_d_model = cnt_d_model
        self.d_model = d_model
        self.height = spatial_size[0]
        self.width = spatial_size[1]

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, cnt_memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     cnt_memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     cnt_memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     cnt_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        ########################## cnt + bg fusion #############################
        tgt = self.bg_norm(tgt)
        cnt_tgt = self.cnt_norm(tgt)

        N = pos.shape[1]

        cnt_tgt = cnt_tgt.permute(1, 2, 0).reshape(N, self.cnt_d_model, self.height, self.width)
        cnt_tgt = cnt_tgt.reshape(N, self.cnt_d_model, self.height * self.width).permute(2, 0, 1)

        # print("memory:", cnt_tgt.shape, pos.shape, cnt_pos.shape, cnt_memory.shape)

        tgt2_cnt, cnt_w = self.cnt_multihead_attn(query=cnt_tgt, #self.with_pos_embed(cnt_tgt, query_pos),
                                                  key=self.with_pos_embed(cnt_memory, cnt_pos),
                                                  value=cnt_memory, attn_mask=cnt_memory_mask,
                                                  key_padding_mask=cnt_memory_key_padding_mask)  # [0]

        # print("cnt_w:", cnt_w.shape, np.unique(cnt_w[0].data.cpu().numpy()))

        # Transfer from 64 channel to 1024 channel
        tgt2_cnt = tgt2_cnt.permute(1, 2, 0).reshape(N, self.cnt_d_model, self.height, self.width)
        tgt2_cnt = tgt2_cnt.reshape(N, self.cnt_d_model * self.height, self.width).permute(2, 0, 1)


        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos), # memory
                                   value=memory, attn_mask=memory_mask, # memory
                                   key_padding_mask=memory_key_padding_mask)[0]

        # overall_tgt = torch.cat([tgt2, tgt2_cnt], dim=-1)

        # tgt = tgt + self.dropout2(
        #     self.activation(tgt2))  # self.dropout2(tgt2) + self.dropout2(tgt2_cnt)
        #########################################################################

        tgt2 = self.norm2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2) + self.dropout3(tgt2_cnt)
        tgt = self.norm3(tgt)
        return tgt, cnt_w

    def forward_pre(self, tgt, memory, cnt_memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # tgt2 = self.norm2(tgt)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                           key=self.with_pos_embed(memory, pos),
        #                           value=memory, attn_mask=memory_mask,
        #                           key_padding_mask=memory_key_padding_mask)[0]
        # tgt = tgt + self.dropout2(tgt2)

        ########################## cnt + bg fusion #############################
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt2_cnt = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(cnt_memory, pos),
                                       value=cnt_memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2) + self.dropout2(tgt2_cnt)
        #########################################################################

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, cnt_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                cnt_memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                cnt_memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                cnt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        # print("memory in post:", memory.shape, pos.shape, cnt_pos.shape)

        if self.normalize_before:
            return self.forward_pre(tgt, memory, cnt_memory, tgt_mask, memory_mask, cnt_memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt,
                                 memory,
                                 cnt_memory,
                                 tgt_mask,
                                 memory_mask,
                                 cnt_memory_mask,
                                 tgt_key_padding_mask,
                                 memory_key_padding_mask,
                                 cnt_memory_key_padding_mask,
                                 pos,
                                 cnt_pos,
                                 query_pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # self.d_model_self = 1024
        # self.d_model = 64

        # self.height = 16
        # self.width = 64

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        # print("tgt:", tgt.shape)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)

        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayer_TP(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.d_model_self = 1024
        self.d_model = d_model

        self.height = 16
        self.width = 64

        self.self_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("pos:", tensor.shape, pos.shape)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        # L, N, C = tgt.shape

        #tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
        #                         key_padding_mask=tgt_key_padding_mask)[0]
        #tgt = tgt + self.dropout1(tgt2)

        tgt2, attn_weights = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask) # q, k, v

        # print("attn_weights:", np.unique(attn_weights[0].data.cpu().numpy()))

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x



class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x

class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)


class ResidualBlock(nn.Module):
    def __init__(self, channels, feat_height=16, feat_width=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.feat_height = feat_height
        self.feat_width = feat_width

    def forward(self, x):
        L, N, C = x.shape

        x = x.permute(1, 2, 0).view(N, C // self.feat_height, self.feat_height, self.feat_width)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return (x + residual).view(N, C, L).permute(2, 0, 1)


class ConvTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, feat_height=16, feat_width=64, text_channels=32):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Implementation of Feedforward model
        # self.ffn = ResidualBlock(d_model // feat_height, feat_height, feat_width)
        # ResidualBlock
        self.ffn = RecurrentResidualBlockTL(d_model // feat_height, text_channels)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.feat_height = feat_height
        self.feat_width = feat_width

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     text_prior: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        ########
        tgt2 = self.ffn(tgt, text_prior, self.feat_height, self.feat_width)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                text_prior: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, text_prior)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

        # self.concat_conv = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, text_emb, feat_height, feat_width):
        # print("text_emb:", text_emb.shape)
        L, N, C = x.shape
        x = x.permute(1, 2, 0).view(N, C // feat_height, feat_height, feat_width)

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # print("text_emb:", text_emb.shape)

        ############ Fusing with TL ############
        cat_feature = torch.cat([residual, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################

        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual).reshape(N, C, L).permute(2, 0, 1)



class ResidualFFBlock(nn.Module):
    def __init__(self, in_channel=16, res_channel=16):
        super(ResidualFFBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, res_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(res_channel)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(res_channel, in_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(res_channel)

    def forward(self, x):
        # x: [L, N, C] -> [N, C, H, W]

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

