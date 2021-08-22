import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data.dataset import get_data
from model.model import Transfollower, lstm_model, nn_model

from config import Settings
settings = Settings()

MODEL = 'transfollower' # ['transfollower','lstm', 'nn']
DATASET = 'SH_shift' # ['SH', 'NGSIM']

exp_name = f'{DATASET}_{MODEL}'
save = f'checkpoints/{exp_name}_model.pt'
writer = SummaryWriter(f'runs/{exp_name}')

# parameters
SEQ_LEN = settings.SEQ_LEN
LABEL_LEN = 40 if MODEL == 'nn' else settings.LABEL_LEN
PRED_LEN = settings.PRED_LEN
BATCH_SIZE = settings.BATCH_SIZE
lr = settings.lr
T = settings.T # data sampling interval
N_EPOCHES = settings.N_EPOCHES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL == 'transfollower':    
    model = Transfollower(enc_in=3).to(device)
elif MODEL == 'lstm':
    model = lstm_model().to(device)
elif MODEL == 'nn':
    model = nn_model().to(device)

model_optim = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

if MODEL == 'nn':
    train_loader, val_loader, _ = get_data(data_name = DATASET, label_len = SEQ_LEN)
else:
    train_loader, val_loader, _ = get_data(data_name= DATASET)


def val(data_loader):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, item in enumerate(data_loader):
            enc_inp = item['his'][:,:, [0,1,4]].float().to(device)

            batch_y = item['svSpd'].float()
            y_label = batch_y[:,-PRED_LEN:,:].to(device)
            batch_y_mark = item['lvSpd'].float().to(device)

            # decoder input
            dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() 
            dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
            dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
            
            # encoder - decoder
            if MODEL == 'nn':
                out = model(dec_inp)
            else:
                out = model(enc_inp, dec_inp)[0]
            
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
        enc_inp = item['his'][:,:, [0,1,4]].float().to(device)

        batch_y = item['svSpd'].float()
        y_label = batch_y[:,-PRED_LEN:,:].to(device)

        batch_y_mark = item['lvSpd'].float().to(device)

        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() 
        dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
        dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
        
        # encoder - decoder
        if MODEL == 'nn':
            out = model(dec_inp)
        else:
            out = model(enc_inp, dec_inp)[0]
        
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

