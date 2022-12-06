import torch
from torch import nn

from model.transformer_torch import Transformer

class Transfollower(nn.Module):
    def __init__(self, config, enc_in = 4, dec_in = 2, d_model = 256, num_encoder_layers = 2, num_decoder_layers = 1):
        super(Transfollower, self).__init__()
        self.transformer = Transformer(d_model= d_model, nhead=8, num_encoder_layers=num_encoder_layers,
                                   num_decoder_layers=num_decoder_layers, dim_feedforward=1024, 
                                   dropout=0.1, activation='relu', custom_encoder=None,
                                   custom_decoder=None, layer_norm_eps=1e-05, batch_first=True, 
                                   device=None, dtype=None)
        self.enc_emb = nn.Linear(enc_in, d_model)
        self.dec_emb = nn.Linear(dec_in, d_model)
        self.out_proj = nn.Linear(d_model, 1, bias = True)
        self.settings = config
        
        self.enc_positional_embedding = nn.Embedding(self.settings.SEQ_LEN, d_model)
        self.dec_positional_embedding = nn.Embedding(self.settings.PRED_LEN + self.settings.LABEL_LEN, d_model)

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
        return out[:,-self.settings.PRED_LEN:,:], enc_attns, dec_attns, enc_dec_attns

MAX_SPD = 25 
class lstm_model(nn.Module):
    def __init__(self, config, input_size = 4, hidden_size = 32, lstm_layers = 1, dropout = 0):
        super(lstm_model, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first = True, dropout = dropout)
        self.decoder = nn.LSTM(2, hidden_size, lstm_layers, batch_first = True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
        nn.init.normal_(self.linear.weight, 0, .02)
        nn.init.constant_(self.linear.bias, 0.0)
        self.settings = config
  
    def forward(self, src, tgt):
        enc_x, (h_n, c_n) = self.encoder(src)
        dec_x, _ = self.decoder(tgt, (h_n, c_n))
        
        out = self.linear(dec_x)
        out = torch.tanh(out)*MAX_SPD/2 + MAX_SPD/2 
        return out[:,-self.settings.PRED_LEN:,:]

# fully connected neural network
class nn_model(nn.Module):
    def __init__(self, config, input_size = 2, hidden_size = 256, dropout = 0.1):
        super(nn_model, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)
            )
        self.settings = config
  
    # using sv speed and lv speed as input. Use 0 as placeholders for future sv speed. 
    def forward(self, src):
        out = self.encoder(src)
        out = torch.tanh(out)*MAX_SPD/2 + MAX_SPD/2 
        return out[:,-self.settings.PRED_LEN:,:]

class Trajectron(nn.Module):
    def __init__(self, config, input_dim = 2) -> None:
        super(Trajectron, self).__init__()
        self.his_encoder = nn.LSTM(input_dim, 32)
        self.interaction_encoder = nn.LSTM(input_dim, 8)
        self.output_layer = nn.Linear(40, 2)
        self.settings = config
    
    def forward(self, inputs, iftest = False):
        """
        inputs: [T, B, N, d] 
        """
        T, B, _, _ = inputs.shape
        outputs = torch.zeros(T, B, 2).cuda()
        obs_length = self.settings.SEQ_LEN

        for framenum in range(T):
            nodes_current = inputs[:framenum + 1] 
        
            if framenum >= obs_length and iftest:
                # Replace ground truth data of SV with prediction part.
                sv_pre = outputs[obs_length - 1:framenum]
                nodes_current[obs_length:, :, -1, :] = sv_pre

            # Only takes the most recent obs_length steps
            if len(nodes_current) > obs_length:
                nodes_current = nodes_current[-obs_length:]

            # encoding sv history
            _, (his_embedding, _) = self.his_encoder(nodes_current[:, :, -1, :])
            his_embedding = his_embedding[-1] # take the hidden state of the last lstm layer

            # encode interaction (here we only have the leading vehicle as the neighbor)
            _, (inter_embedding, _) = self.interaction_encoder(nodes_current[:, :, 0, :])
            inter_embedding = inter_embedding[-1] # take the last layer

            # concat the two embeddings
            comb_embedding = torch.cat([his_embedding, inter_embedding], axis = -1)
            
            outputs_current = self.output_layer(comb_embedding) # B, d
            outputs[framenum] = outputs_current

        return outputs
