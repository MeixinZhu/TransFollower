import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from data.dataset import get_data_star
from model.star.star import STAR

from config import Settings, HighDSettings
DATASET = 'highD' # ['SH', 'NGSIM', 'highD']
if DATASET == 'highD':
    settings = HighDSettings()
else:
    settings = Settings()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

MODEL = 'star' 
exp_name = f'{DATASET}_{MODEL}'
save = f'checkpoints/{exp_name}_model.pt'
writer = SummaryWriter(f'runs/{exp_name}')

# parameters
SEQ_LEN = settings.SEQ_LEN
if MODEL == 'nn':
    settings.LABEL_LEN = SEQ_LEN
LABEL_LEN = settings.LABEL_LEN
PRED_LEN = settings.PRED_LEN
settings.BATCH_SIZE = 8
lr = settings.lr
T = settings.T # data sampling interval
N_EPOCHES = settings.N_EPOCHES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = STAR(config = settings).to(device)

model_optim = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

train_loader, val_loader, _ = get_data_star(data_name = DATASET, config = settings)

def val(data_loader):
    print('Evaluating STAR model.')
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, item in tqdm(enumerate(data_loader)):
            item = item.float().to(device)
            item = item.transpose(0, 1) # T, B, N, d
            out = model(item[:-1], iftest = True)
            
            label = item[1:] # use :t-1 data to predict t time step data
            svSpd_ = out[..., -1] # speed prediction for SV
            svSpd_obs = label[..., -1, -1] 
            svPos_ = out[..., 0] # spacing prediction for SV
            svPos_obs = label[..., -1, 0]
            lvPos = label[..., 0, 0]
            svSpacing_ = lvPos - svPos_
            svSpacing_obs = lvPos - svPos_obs
            
            loss = criterion(svSpacing_[SEQ_LEN:], svSpacing_obs[SEQ_LEN:]) + criterion(svSpd_[SEQ_LEN:], svSpd_obs[SEQ_LEN:])

            total_loss.append(loss.item())
    model.train()
    return np.mean(total_loss)

# train
best_val_loss = None
model.train() 

for epoch in range(N_EPOCHES):
    train_losses = []
    for i, item in tqdm(enumerate(train_loader)):
        item = item.float().to(device)
        item = item.transpose(0, 1) # T, B, N, d
        out = model(item[:-1])
        label = item[1:, :, -1, :] # [lv, sv] for N
        loss = criterion(out, label) 

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

