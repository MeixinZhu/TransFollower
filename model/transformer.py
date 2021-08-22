
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
if not '..' in sys.path:
    sys.path += ['..']
from config import Settings
settings = Settings()

class SelfAttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.Q = nn.Linear(d_model, d_model)

        nn.init.normal_(self.K.weight, 0, .02)
        nn.init.normal_(self.Q.weight, 0, .02)
        nn.init.normal_(self.V.weight, 0, .02)


    def forward(self, x, padding_mask = None, subsq_mask = None):
        # src shape: [N, SEQ, D_MODEL]

        #SelfAttention:
        keys = self.K.forward(x)
        values = self.V.forward(x)
        queries = self.Q.forward(x)

        sqrt_d = self.d_model ** 0.5

        att = torch.matmul(queries, keys.transpose(1,2)) / sqrt_d
        # att shape: [N, SEQ, SEQ]
        # Broadcast padding mask to word attentions so that word attention does not attend to positions outside the sentence
        if padding_mask is not None:
            att = att + padding_mask.transpose(1,2)
        # Add subsequent mask so that each position can attend only itself and the previous elements
        if subsq_mask is not None:
            att = att + subsq_mask.unsqueeze(0)

        att_softmax = torch.softmax(att, dim=2)
        # shape: [N, SEQ, SEQ]
        att_out = torch.matmul(att_softmax, values)
        # shape: [N, SEQ, D_MODEL]
        # pdb.set_trace()

        return att_out, keys, values

class MemAttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.Q = nn.Linear(d_model, d_model)

        nn.init.normal_(self.Q.weight, 0, .02)

    def forward(self, x, mem_padding_mask, keys = None, values = None):

        # X shape: [N, SEQ, D_MODEL]

        queries = self.Q.forward(x)

        sqrt_d = self.d_model ** 0.5

        att = torch.matmul(queries, keys.transpose(1,2)) / sqrt_d
        # att shape: [N, SEQ_TGT, SEQ_SRC]

        # Broadcast padding mask to word attentions so that word attention does not attend to positions outside the source sentence
        if mem_padding_mask is not None:
            att = att + mem_padding_mask.transpose(1,2)

        att_softmax = torch.softmax(att, dim=2)
        # shape: [N, SEQ_TGT, SEQ_SRC]
        att_out = torch.matmul(att_softmax, values)
        # shape: [N, SEQ_TGT, D_MODEL]

        return att_out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(d_model) for i in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_model, d_model)

        nn.init.normal_(self.linear.weight, 0, .02)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, src, src_padding_mask, src_subsq_mask):

        out_cat = None
        keys = None
        values = None

        for i in range(self.num_heads):
            head_outp, keys, values = self.heads[i].forward(src, src_padding_mask, src_subsq_mask)

            if i == 0:
                out_cat = head_outp
            else:
                out_cat = torch.cat([out_cat, head_outp], dim=2)

        ret = self.linear.forward(out_cat)

        return ret, keys, values


class MultiHeadMemAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.heads = nn.ModuleList([MemAttentionHead(d_model) for i in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_model, d_model)
        nn.init.normal_(self.linear.weight, 0, .02)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, src, src_padding_mask, keys, values):

        out_cat = None
        for i in range(self.num_heads):
            head_outp = self.heads[i].forward(src, src_padding_mask, keys = keys, values = values)

            if i == 0:
                out_cat = head_outp
            else:
                out_cat = torch.cat([out_cat, head_outp], dim=2)

        ret = self.linear.forward(out_cat)

        return ret

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_att_heads, ff_dim = 1024, dropout = 0.1):
        super().__init__()

        self.multihead_attention = MultiHeadSelfAttention(d_model, num_att_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.att_sublayer_norm = torch.nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.dropout_lin = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lin_sublayer_norm = torch.nn.LayerNorm(d_model)

        nn.init.normal_(self.linear1.weight, 0, .02)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.normal_(self.linear2.weight, 0, .02)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, src, src_padding_mask, src_subsq_mask):

        residual_1 = src
        x, keys, values = self.multihead_attention.forward(src, src_padding_mask, src_subsq_mask)
        x = self.att_sublayer_norm.forward(residual_1 + self.dropout1(x))

        residual_2 = x
        x = self.linear2(self.dropout_lin(self.relu(self.linear1.forward(x))))
        x = self.lin_sublayer_norm(residual_2 + self.dropout2(x))

        return x, keys, values


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_att_heads, ff_dim = 1024, dropout = 0.1):
        super().__init__()

        self.multihead_self_attention = MultiHeadSelfAttention(d_model, num_att_heads)
        self.self_att_sublayer_norm = torch.nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.multihead_mem_attention = MultiHeadMemAttention(d_model, num_att_heads)
        self.mem_att_sublayer_norm = torch.nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout_lin = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.lin_sublayer_norm = torch.nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        nn.init.normal_(self.linear1.weight, 0, .02)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.normal_(self.linear2.weight, 0, .02)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x, src_padding_mask, tgt_padding_mask, tgt_subsq_mask, mem_keys, mem_values):

        residual_1 = x
        x, keys, values = self.multihead_self_attention.forward(x, tgt_padding_mask, tgt_subsq_mask)
        x = self.self_att_sublayer_norm.forward(residual_1 + self.dropout1(x))

        residual_2 = x
        x = self.multihead_mem_attention.forward(x, src_padding_mask, keys = mem_keys, values = mem_values)
        x = self.mem_att_sublayer_norm.forward(residual_2 + self.dropout2(x))

        residual_3 = x
        x = self.linear2(self.dropout_lin(self.relu(self.linear1.forward(x))))
        x = self.lin_sublayer_norm(residual_3 + self.dropout3(x))

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_att_heads) for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self,src, src_padding_mask, src_subsq_mask):
        x = src

        keys = None
        values = None
        for layer in self.layers:
            x, keys, values = layer.forward(x, src_padding_mask, src_subsq_mask)

        x = self.norm.forward(x)

        return keys, values


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_att_heads) for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, src_padding_mask, tgt_padding_mask, tgt_subsq_mask, mem_keys, mem_values):
        x = tgt
        for layer in self.layers:
            x = layer.forward(x, src_padding_mask, tgt_padding_mask, tgt_subsq_mask, mem_keys, mem_values)

        x = self.norm.forward(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.sin_args = torch.zeros(1, self.d_model)
        self.cos_args = torch.zeros(1, self.d_model)
        for i in range(self.d_model//2):
            self.sin_args[0,i * 2] = 1000**(2.*i/self.d_model)
            self.cos_args[0,i * 2 + 1] = 1000**(2.*i/self.d_model)

        self.sin_args_filled = (self.sin_args > 1e-10).float()
        self.sin_args = self.sin_args + (self.sin_args < 1e-10).float()

        self.cos_args_filled = (self.cos_args > 1e-10).float()
        self.cos_args = self.cos_args + (self.cos_args < 1e-10).float()

    def forward(self, x):
        for pos in range(x.size()[-2]):
            x[:,pos,:] = x[:,pos,:] + \
                         torch.sin(pos / self.sin_args.to(x.device)) * self.sin_args_filled.to(x.device) + \
                         torch.cos(pos / self.cos_args.to(x.device)) * self.cos_args_filled.to(x.device)
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_att_heads, src_dim, tgt_dim, dropoutio = 0.0):
        super().__init__()

        self.src_emb = nn.Linear(src_dim, d_model)
        self.tgt_emb = nn.Linear(tgt_dim, d_model)

        self.positional_encoder = PositionalEncoding(d_model)
        self.encoder = Encoder(num_layers, d_model, num_att_heads)
        self.decoder = Decoder(num_layers, d_model, num_att_heads)
        self.dropi = nn.Dropout(dropoutio)
        self.dropo = nn.Dropout(dropoutio)

        self.outp_proj = nn.Linear(d_model, 1)

        nn.init.normal_(self.src_emb.weight, 0, .02)
        nn.init.constant_(self.src_emb.bias, 0.0)
        nn.init.normal_(self.tgt_emb.weight, 0, .02)
        nn.init.constant_(self.tgt_emb.bias, 0.0)

        nn.init.normal_(self.outp_proj.weight, 0, .02)
        nn.init.constant_(self.outp_proj.bias, 0.0)

    def forward(self, src, tgt, src_padding_mask = None, src_subsq_mask = None, tgt_padding_mask = None, tgt_subsq_mask = None):
        enc_x = self.src_emb.forward(src)
        enc_x = self.dropi(self.positional_encoder.forward(enc_x))
        enc_keys, enc_values = self.encoder.forward(enc_x, src_padding_mask, src_subsq_mask)

        dec_x = self.tgt_emb.forward(tgt)
        dec_x = self.positional_encoder.forward(dec_x)
        dec_x = self.decoder.forward(dec_x, src_padding_mask, tgt_padding_mask, tgt_subsq_mask, enc_keys, enc_values)
        dec_x = self.dropo(dec_x)

        out = self.outp_proj(dec_x)
        return out[:,-settings.PRED_LEN:,:]

def get_square_subsequent_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_padding_mask(input, val1 = float('-inf'), val2 = float(0.0)):
    mask = torch.ones(input.size()).to(input.device)
    mask = mask.float().masked_fill(input == 0, val1).masked_fill(input > 0, val2)
    return mask


def get_one_hot(x, out_dim, mask):

    tens = x.view(-1)
    tens_one_hot = torch.zeros(list(tens.size()) + [out_dim]).to(x.device)
    for i in range(len(tens)):
        tens_one_hot[i,tens[i]] = 1

    tens_one_hot = tens_one_hot.view(list(x.size()) + [out_dim])
    tens_one_hot = tens_one_hot * mask
    return tens_one_hot.to(x.device)