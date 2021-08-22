import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data.dataset import get_data_pretrain
from model.model import Transfollower_pretrain

from config import Settings
settings = Settings()

MODEL = 'transfollower' 
DATASET = 'SH' # ['SH', 'NGSIM']
exp_name = f'{DATASET}_{MODEL}_pretrain'
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
   
model = Transfollower_pretrain().to(device)

model_optim = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss() #nn.MSELoss()

train_loader, val_loader = get_data_pretrain(data_name= DATASET)

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def val(data_loder):
    model.eval()
    total_loss = []
    f1_scores = []
    with torch.no_grad():
        for i, item in enumerate(data_loder):
            enc_inp = item['his'].float().to(device)

            batch_y = item['svSpd'].float()
            y_label = batch_y[:,-PRED_LEN:,:].to(device)
            batch_y_mark = item['lvSpd'].float().to(device)
            label = item['label'].long().to(device)

            # decoder input
            dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() + \
                    batch_y[:,:LABEL_LEN,:].mean(axis = 1, keepdim=True)
            dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
            dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
            
            # encoder - decoder
            out = model(enc_inp, dec_inp)
            
            # lvSpd, spacing = item['lvSpd'][:, LABEL_LEN:,:].float().to(device), item['spacing'].float().to(device)
            # relSpd_ = (lvSpd - out).squeeze()
            # spacing_ = torch.cumsum(T*(relSpd_[:,:-1] + relSpd_[:,1:])/2, dim = -1) + item['s0'].float().to(device)

            # loss = criterion(out, y_label) + criterion(spacing_, spacing)
            loss = criterion(out, label)

            _, pred = torch.max(out, dim=1)
            f1_score = f1_loss(label, pred).item()
            f1_scores.append(f1_score)

            total_loss.append(loss.item())

    model.train()
    return np.mean(total_loss), np.mean(f1_scores)

# train
best_val_loss = None
model.train()
for epoch in range(N_EPOCHES):
    train_losses = []

    f1_scores = []

    for i, item in enumerate(train_loader):
        enc_inp = item['his'].float().to(device)

        batch_y = item['svSpd'].float()
        y_label = batch_y[:,-PRED_LEN:,:].to(device)
        batch_y_mark = item['lvSpd'].float().to(device)
        label = item['label'].long().to(device)

        # decoder input
        dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() + \
                batch_y[:,:LABEL_LEN,:].mean(axis = 1, keepdim=True)
        dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
        dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
        
        # encoder - decoder
        out = model(enc_inp, dec_inp)
        
        # lvSpd, spacing = item['lvSpd'][:, LABEL_LEN:,:].float().to(device), item['spacing'].float().to(device)
        # relSpd_ = (lvSpd - out).squeeze()
        # spacing_ = torch.cumsum(T*(relSpd_[:,:-1] + relSpd_[:,1:])/2, dim = -1) + item['s0'].float().to(device)
        # loss = criterion(out, y_label) + criterion(spacing_, spacing)

        _, pred = torch.max(out, dim=1)
        loss = criterion(out, label)

        model_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        model_optim.step()

        train_losses.append(loss.item())
        f1_score = f1_loss(label, pred).item()
        f1_scores.append(f1_score)

    train_loss = np.mean(train_losses)
    train_f1 = np.mean(f1_scores)
    val_loss, val_f1 = val(val_loader)
    
    if not best_val_loss or best_val_loss > val_loss:
        with open(save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss

    print("Epoch: {0}| Train Loss: {1:.7f} Vali Loss: {2:.7f} Best val loss: {3:.7f}".format(
                epoch + 1, train_loss, val_loss, best_val_loss))
    print(f'Train f1:{train_f1}, val f1:{val_f1}')
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/vali', val_loss, epoch)
    writer.add_scalar('F1/train', train_f1, epoch)
    writer.add_scalar('F1/vali', val_f1, epoch)
    writer.close()

