import torch
from torch import nn

import sys
if not '..' in sys.path:
    sys.path += ['..']
from config import Settings
settings = Settings()

from model.transformer_torch import Transformer

class Transfollower(nn.Module):
    def __init__(self, enc_in = 4, dec_in = 2, d_model = 256):
        super(Transfollower, self).__init__()
        self.transformer = Transformer(d_model= d_model, nhead=8, num_encoder_layers=2,
                                   num_decoder_layers=1, dim_feedforward=1024, 
                                   dropout=0.1, activation='relu', custom_encoder=None,
                                   custom_decoder=None, layer_norm_eps=1e-05, batch_first=True, 
                                   device=None, dtype=None)
        self.enc_emb = nn.Linear(enc_in, d_model)
        self.dec_emb = nn.Linear(dec_in, d_model)
        self.out_proj = nn.Linear(d_model, 1, bias = True)
        
        self.enc_positional_embedding = nn.Embedding(settings.SEQ_LEN, d_model)
        self.dec_positional_embedding = nn.Embedding(settings.PRED_LEN + settings.LABEL_LEN, d_model)

        nn.init.normal_(self.enc_emb.weight, 0, .02)
        nn.init.normal_(self.dec_emb.weight, 0, .02)
        nn.init.normal_(self.out_proj.weight, 0, .02)
        nn.init.normal_(self.enc_positional_embedding.weight, 0, .02)
        nn.init.normal_(self.dec_positional_embedding.weight, 0, .02)

    def forward(self, enc_inp, dec_inp):
        enc_pos = torch.arange(0, enc_inp.shape[1], dtype=torch.long).to(enc_inp.device)
        dec_pos = torch.arange(0, dec_inp.shape[1], dtype=torch.long).to(dec_inp.device)
        enc_inp = self.enc_emb(enc_inp) + self.enc_positional_embedding(enc_pos)[None,:,:]
        dec_inp = self.dec_emb(dec_inp) + self.dec_positional_embedding(dec_pos)[None,:,:]
        
        out, enc_attns, dec_attns, enc_dec_attns = self.transformer(enc_inp, dec_inp)
        out = self.out_proj(out)
        return out[:,-settings.PRED_LEN:,:], enc_attns, dec_attns, enc_dec_attns

MAX_SPD = 25 
class lstm_model(nn.Module):
    def __init__(self, input_size = 4, hidden_size = 256, lstm_layers = 2, dropout = 0.1):
        super(lstm_model, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first = True, dropout = dropout)
        self.decoder = nn.LSTM(2, hidden_size, lstm_layers, batch_first = True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
        nn.init.normal_(self.linear.weight, 0, .02)
        nn.init.constant_(self.linear.bias, 0.0)
  
    def forward(self, src, tgt):
        enc_x, (h_n, c_n) = self.encoder(src)
        dec_x, _ = self.decoder(tgt, (h_n, c_n))
        
        out = self.linear(dec_x)
        out = torch.tanh(out)*MAX_SPD/2 + MAX_SPD/2 
        return out[:,-settings.PRED_LEN:,:]

# fully connected neural network
class nn_model(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 256, dropout = 0.1):
        super(nn_model, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)
            )
  
    # using sv speed and lv speed as input. Use 0 as placeholders for future sv speed. 
    def forward(self, src):
        out = self.encoder(src)
        out = torch.tanh(out)*MAX_SPD/2 + MAX_SPD/2 
        return out[:,-settings.PRED_LEN:,:]

class Transfollower_pretrain(nn.Module):
    def __init__(self, enc_in = 3, dec_in = 2, d_model = 256):
        super(Transfollower_pretrain, self).__init__()
        self.transformer = nn.Transformer(d_model= d_model, nhead=8, num_encoder_layers=2,
                                   num_decoder_layers=1, dim_feedforward=1024, 
                                   dropout=0.1, activation='relu', custom_encoder=None,
                                   custom_decoder=None, layer_norm_eps=1e-05, batch_first=True, 
                                   device=None, dtype=None)
        self.enc_emb = nn.Linear(enc_in, d_model)
        self.dec_emb = nn.Linear(dec_in, d_model)
        self.out_proj = nn.Linear(d_model, 1, bias = True)
        self.cls_head = nn.Linear(d_model, 2) 
        
        self.enc_positional_embedding = nn.Embedding(settings.SEQ_LEN, d_model)
        self.dec_positional_embedding = nn.Embedding(settings.PRED_LEN + settings.LABEL_LEN, d_model)

        nn.init.normal_(self.enc_emb.weight, 0, .02)
        nn.init.normal_(self.dec_emb.weight, 0, .02)
        nn.init.normal_(self.out_proj.weight, 0, .02)
        nn.init.normal_(self.enc_positional_embedding.weight, 0, .02)
        nn.init.normal_(self.dec_positional_embedding.weight, 0, .02)
        nn.init.normal_(self.cls_head.weight, 0, .02)

    def forward(self, enc_inp, dec_inp, task = 'cls'):
        enc_pos = torch.arange(0, enc_inp.shape[1], dtype=torch.long).to(enc_inp.device)
        dec_pos = torch.arange(0, dec_inp.shape[1], dtype=torch.long).to(dec_inp.device)
        enc_inp = self.enc_emb(enc_inp) + self.enc_positional_embedding(enc_pos)[None,:,:]
        dec_inp = self.dec_emb(dec_inp) + self.dec_positional_embedding(dec_pos)[None,:,:]
        
        out = self.transformer(enc_inp, dec_inp)
        
        if task == 'reg':
            out = self.out_proj(out)
            return out[:,-settings.PRED_LEN:,:]
        else:
            out = self.cls_head(out[:,-1,:].squeeze(1))
            return torch.log_softmax(out, dim = -1)


# # below is a revised version of transformer based on pytorch official
# import copy
# from typing import Optional, Any

# import torch
# from torch import Tensor
# from torch.nn import functional as F
# from torch.nn import Module
# from torch.nn import MultiheadAttention
# from torch.nn import ModuleList
# from torch.nn.init import xavier_uniform_
# from torch.nn import Dropout
# from torch.nn import Linear
# from torch.nn import LayerNorm, BatchNorm1d


# class Transformer(Module):
#     r"""A transformer model. User is able to modify the attributes as needed. The architecture
#     is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
#     Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
#     Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
#     Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
#     model with corresponding parameters.

#     Args:
#         d_model: the number of expected features in the encoder/decoder inputs (default=512).
#         nhead: the number of heads in the multiheadattention models (default=8).
#         num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
#         num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
#         custom_encoder: custom encoder (default=None).
#         custom_decoder: custom decoder (default=None).
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

#     Examples::
#         >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
#         >>> src = torch.rand((10, 32, 512))
#         >>> tgt = torch.rand((20, 32, 512))
#         >>> out = transformer_model(src, tgt)

#     Note: A full example to apply nn.Transformer module for the word language model is available in
#     https://github.com/pytorch/examples/tree/master/word_language_model
#     """

#     def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
#                  num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(Transformer, self).__init__()

#         if custom_encoder is not None:
#             self.encoder = custom_encoder
#         else:
#             encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
#                                                     activation, layer_norm_eps, batch_first,
#                                                     **factory_kwargs)
#             encoder_norm = BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)#LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#             self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

#         if custom_decoder is not None:
#             self.decoder = custom_decoder
#         else:
#             decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
#                                                     activation, layer_norm_eps, batch_first,
#                                                     **factory_kwargs)
#             decoder_norm = BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs) #LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#             self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

#         self._reset_parameters()

#         self.d_model = d_model
#         self.nhead = nhead

#         self.batch_first = batch_first

#     def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Take in and process masked source/target sequences.

#         Args:
#             src: the sequence to the encoder (required).
#             tgt: the sequence to the decoder (required).
#             src_mask: the additive mask for the src sequence (optional).
#             tgt_mask: the additive mask for the tgt sequence (optional).
#             memory_mask: the additive mask for the encoder output (optional).
#             src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
#             tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
#             memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

#         Shape:
#             - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
#             - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
#             - src_mask: :math:`(S, S)`.
#             - tgt_mask: :math:`(T, T)`.
#             - memory_mask: :math:`(T, S)`.
#             - src_key_padding_mask: :math:`(N, S)`.
#             - tgt_key_padding_mask: :math:`(N, T)`.
#             - memory_key_padding_mask: :math:`(N, S)`.

#             Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
#             positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
#             while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
#             are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
#             is provided, it will be added to the attention weight.
#             [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
#             the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
#             positions will be unchanged. If a BoolTensor is provided, the positions with the
#             value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

#             - output: :math:`(T, N, E)`, `(N, T, E)` if batch_first.

#             Note: Due to the multi-head attention architecture in the transformer model,
#             the output sequence length of a transformer is same as the input sequence
#             (i.e. target) length of the decode.

#             where S is the source sequence length, T is the target sequence length, N is the
#             batch size, E is the feature number

#         Examples:
#             >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
#         """

#         if not self.batch_first and src.size(1) != tgt.size(1):
#             raise RuntimeError("the batch number of src and tgt must be equal")
#         elif self.batch_first and src.size(0) != tgt.size(0):
#             raise RuntimeError("the batch number of src and tgt must be equal")

#         if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
#             raise RuntimeError("the feature number of src and tgt must be equal to d_model")

#         memory, enc_attns = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
#         output, dec_attns = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
#                               tgt_key_padding_mask=tgt_key_padding_mask,
#                               memory_key_padding_mask=memory_key_padding_mask)
#         return output, enc_attns, dec_attns


#     def generate_square_subsequent_mask(self, sz: int) -> Tensor:
#         r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
#             Unmasked positions are filled with float(0.0).
#         """
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask


#     def _reset_parameters(self):
#         r"""Initiate parameters in the transformer model."""

#         for p in self.parameters():
#             if p.dim() > 1:
#                 xavier_uniform_(p)



# class TransformerEncoder(Module):
#     r"""TransformerEncoder is a stack of N encoder layers

#     Args:
#         encoder_layer: an instance of the TransformerEncoderLayer() class (required).
#         num_layers: the number of sub-encoder-layers in the encoder (required).
#         norm: the layer normalization component (optional).

#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = transformer_encoder(src)
#     """
#     __constants__ = ['norm']

#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoder, self).__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layers in turn.

#         Args:
#             src: the sequence to the encoder (required).
#             mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         output = src
#         attns = []
#         for mod in self.layers:
#             output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             attns.append(attn)

#         if self.norm is not None:
#             output = self.norm(output.permute(0,2,1)).permute(0,2,1)

#         return output, attns



# class TransformerDecoder(Module):
#     r"""TransformerDecoder is a stack of N decoder layers

#     Args:
#         decoder_layer: an instance of the TransformerDecoderLayer() class (required).
#         num_layers: the number of sub-decoder-layers in the decoder (required).
#         norm: the layer normalization component (optional).

#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = transformer_decoder(tgt, memory)
#     """
#     __constants__ = ['norm']

#     def __init__(self, decoder_layer, num_layers, norm=None):
#         super(TransformerDecoder, self).__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer in turn.

#         Args:
#             tgt: the sequence to the decoder (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         output = tgt
#         attns = []
#         for mod in self.layers:
#             output, attn = mod(output, memory, tgt_mask=tgt_mask,
#                          memory_mask=memory_mask,
#                          tgt_key_padding_mask=tgt_key_padding_mask,
#                          memory_key_padding_mask=memory_key_padding_mask)
#             attns.append(attn)

#         if self.norm is not None:
#             output = self.norm(output.permute(0,2,1)).permute(0,2,1)

#         return output, attns


# class TransformerEncoderLayer(Module):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.

#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False``.

#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)

#     Alternatively, when ``batch_first`` is ``True``:
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> src = torch.rand(32, 10, 512)
#         >>> out = encoder_layer(src)
#     """
#     __constants__ = ['batch_first']

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
#                  layer_norm_eps=1e-5, batch_first=False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

#         self.norm1 = BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)#LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)#LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)

#         self.activation = _get_activation_fn(activation)

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerEncoderLayer, self).__setstate__(state)

#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.

#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src.permute(0,2,1)).permute(0,2,1)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src.permute(0,2,1)).permute(0,2,1)
#         return src, attn



# class TransformerDecoderLayer(Module):
#     r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
#     This standard decoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.

#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False``.

#     Examples::
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
#         >>> memory = torch.rand(10, 32, 512)
#         >>> tgt = torch.rand(20, 32, 512)
#         >>> out = decoder_layer(tgt, memory)

#     Alternatively, when ``batch_first`` is ``True``:
#         >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> memory = torch.rand(32, 10, 512)
#         >>> tgt = torch.rand(32, 20, 512)
#         >>> out = decoder_layer(tgt, memory)
#     """
#     __constants__ = ['batch_first']

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
#                  layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                                  **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

#         self.norm1 = BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)#LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)#LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm3 = BatchNorm1d(d_model, eps=layer_norm_eps, **factory_kwargs)#LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#         self.dropout3 = Dropout(dropout)

#         self.activation = _get_activation_fn(activation)

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerDecoderLayer, self).__setstate__(state)

#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the inputs (and mask) through the decoder layer.

#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_mask: the mask for the memory sequence (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
#             memory_key_padding_mask: the mask for the memory keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         tgt2, attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt.permute(0,2,1)).permute(0,2,1)
#         tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt.permute(0,2,1)).permute(0,2,1)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt.permute(0,2,1)).permute(0,2,1)
#         return tgt, attn



# def _get_clones(module, N):
#     return ModuleList([copy.deepcopy(module) for i in range(N)])


# def _get_activation_fn(activation):
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu

#     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))