import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data.dataset import get_data
from model.model import Transfollower, lstm_model, nn_model
from config import Settings, HighDSettings
# import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=""  # specify which GPU(s) to be used


DATASET = 'highD' # ['SH', 'NGSIM', 'highD']
if DATASET == 'highD':
    settings = HighDSettings()
else:
    setting = Settings()

MODEL = 'lstm' # ['transfollower','lstm', 'nn']

exp_name = f'{DATASET}_{MODEL}'
save = f'checkpoints/{exp_name}_model.pt'
writer = SummaryWriter(f'runs/{exp_name}')

# parameters
SEQ_LEN = settings.SEQ_LEN

if MODEL == 'nn':
    settings.LABEL_LEN = SEQ_LEN

LABEL_LEN = settings.LABEL_LEN
PRED_LEN = settings.PRED_LEN
BATCH_SIZE = settings.BATCH_SIZE
lr = settings.lr
T = settings.T # data sampling interval
N_EPOCHES = settings.N_EPOCHES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if MODEL == 'transfollower':    
    model = Transfollower(config = settings).to(device)
elif MODEL == 'lstm':
    model = lstm_model(config = settings).to(device)
elif MODEL == 'nn':
    model = nn_model(config = settings).to(device)

model_optim = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

train_loader, val_loader, _ = get_data(data_name = DATASET, config = settings)

def val(data_loader):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, item in enumerate(data_loader):
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
            if MODEL == 'nn':
                out = model(dec_inp)
            elif MODEL == 'transfollower':
                out = model(enc_inp, dec_inp)[0]
            else:
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
                batch_y[:,:LABEL_LEN,:].mean(axis = 1, keepdim=True) # Use mean padding instead of zero padding
        dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device) # concat label with padding
        dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
        
        # encoder - decoder
        if MODEL == 'nn':
            out = model(dec_inp)
        elif MODEL == 'transfollower':
            out = model(enc_inp, dec_inp)[0]
        else:
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
    
    if not best_val_loss or best_val_loss > val_loss:
        with open(save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss

    print("Epoch: {0}| Train Loss: {1:.7f} Vali Loss: {2:.7f} Best val loss: {3:.7f}".format(
                epoch + 1, train_loss, val_loss, best_val_loss))
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/vali', val_loss, epoch)
    writer.close()

