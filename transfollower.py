import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import shutil
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

# parameters
SEQ_LEN = 40
LABEL_LEN = 10
PRED_LEN = 150 - SEQ_LEN
BATCH_SIZE = 512
lr = 6e-5
T = 0.1 # data sampling interval
N_EPOCHES = 1000

exp_name = 'informer_SHNDS_full'
save = f'checkpoints/{exp_name}_model.pt'
writer = SummaryWriter(f'runs/{exp_name}')
# shutil.rmtree(f'runs/{exp_name}')

class SHNDS(torch.utils.data.Dataset):
    def __init__(self, root = '/media/meixin/transfollower/data', seq_len = 10, 
                 label_len =10, split = 'train_SH_sample'):
        self.seq_len = seq_len
        self.pred_len = 150 - self.seq_len
        self.label_len = label_len
        self.data = np.load(os.path.join(root, split + '.npy'), allow_pickle = True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        event = self.data[idx]
        ret = dict()
        ret['his'] = torch.from_numpy(event[:self.seq_len])
        ret['lvSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 3])).unsqueeze(-1) 
        ret['svSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 1])).unsqueeze(-1) 
        ret['spacing'] = torch.from_numpy(np.array(event[self.seq_len+1:150, 0])) 
        ret['s0'] = torch.from_numpy(np.array(event[self.seq_len, 0])).unsqueeze(-1)
        return ret

SAMPLE = False
if not SAMPLE:
    train_dataset = SHNDS(seq_len = SEQ_LEN, split = 'train_SH', label_len = LABEL_LEN)
    val_dataset = SHNDS(seq_len = SEQ_LEN, split = 'val_SH', label_len = LABEL_LEN)
    test_dataset = SHNDS(seq_len = SEQ_LEN, split = 'test_SH', label_len = LABEL_LEN)
else:
    train_dataset = SHNDS(seq_len = SEQ_LEN, split = 'train_SH_sample', label_len = LABEL_LEN)
    val_dataset = SHNDS(seq_len = SEQ_LEN, split = 'val_SH_sample', label_len = LABEL_LEN)
    test_dataset = SHNDS(seq_len = SEQ_LEN, split = 'test_SH_sample', label_len = LABEL_LEN)
print('Total data samples for train, val, and test')
print(len(train_dataset), len(val_dataset), len(test_dataset))

drop_last = False
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=drop_last)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=drop_last)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=drop_last)

print(f'Total batches for train, val, and test with batch size {BATCH_SIZE}')
print(len(train_loader), len(val_loader), len(test_loader))

class Transfollower(nn.Module):
    def __init__(self, enc_in = 4, dec_in = 2, d_model = 256):
        super(Transfollower, self).__init__()
        self.transformer = nn.Transformer(d_model= d_model, nhead=8, num_encoder_layers=2,
                                   num_decoder_layers=1, dim_feedforward=1024, 
                                   dropout=0.1, activation='relu', custom_encoder=None,
                                   custom_decoder=None, layer_norm_eps=1e-05, batch_first=True, 
                                   device=None, dtype=None)
        self.enc_emb = nn.Linear(enc_in, d_model)
        self.dec_emb = nn.Linear(dec_in, d_model)
        self.out_proj = nn.Linear(d_model, 1, bias = True)
        
        self.enc_positional_embedding = nn.Embedding(SEQ_LEN, d_model)
        self.dec_positional_embedding = nn.Embedding(PRED_LEN + LABEL_LEN, d_model)

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
        
        out = self.transformer(enc_inp, dec_inp)
        out = self.out_proj(out)
        return out[:,-PRED_LEN:,:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transfollower().to(device)
model_optim = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

def val(data_loder):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, item in enumerate(data_loder):
            enc_inp = item['his'].float().to(device)

            batch_y = item['svSpd'].float()
            y_label = batch_y[:,-PRED_LEN:,:].to(device)
            batch_y_mark = item['lvSpd'].float().to(device)
            # decoder input
            dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() + \
                    batch_y[:,:LABEL_LEN,:].mean(axis = 1, keepdim=True)
            dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
            dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
            
            # encoder - decoder
            out = model(enc_inp, dec_inp)
            
            lvSpd, spacing = item['lvSpd'][:, LABEL_LEN:,:].float().to(device), item['spacing'].float().to(device)
            relSpd_ = (lvSpd - out).squeeze()
            spacing_ = torch.cumsum(T*(relSpd_[:,:-1] + relSpd_[:,1:])/2, dim = -1) + item['s0'].float().to(device)
            loss = criterion(out, y_label) + criterion(spacing_, spacing)

            total_loss.append(loss.item())
    model.train()
    return np.mean(total_loss)

# train
best_val_loss = None
model.train()
for epoch in range(N_EPOCHES):
    train_losses = []
    for i, item in enumerate(train_loader):
        enc_inp = item['his'].float().to(device)

        batch_y = item['svSpd'].float()
        y_label = batch_y[:,-PRED_LEN:,:].to(device)
        batch_y_mark = item['lvSpd'].float().to(device)
        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() + \
                batch_y[:,:LABEL_LEN,:].mean(axis = 1, keepdim=True)
        dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
        dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
        
        # encoder - decoder
        out = model(enc_inp, dec_inp)
        
        lvSpd, spacing = item['lvSpd'][:, LABEL_LEN:,:].float().to(device), item['spacing'].float().to(device)
        relSpd_ = (lvSpd - out).squeeze()
        spacing_ = torch.cumsum(T*(relSpd_[:,:-1] + relSpd_[:,1:])/2, dim = -1) + item['s0'].float().to(device)
        loss = criterion(out, y_label) + criterion(spacing_, spacing)
        model_optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        model_optim.step()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = val(val_loader)
    print("Epoch: {0}| Train Loss: {1:.7f} Vali Loss: {2:.7f} Best val loss: {3:.7f}".format(
                epoch + 1, train_loss, val_loss, best_val_loss))

    if not best_val_loss or best_val_loss > val_loss:
        with open(save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/vali', val_loss, epoch)
    writer.close()

