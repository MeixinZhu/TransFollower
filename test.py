import torch
from torch import nn
import numpy as np

from data.dataset import get_data
from model.model import Transfollower, lstm_model, nn_model

from config import Settings
settings = Settings()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

def val(model_name, data_name):
    # parameters
    SEQ_LEN = settings.SEQ_LEN
    LABEL_LEN = 40 if model_name == 'nn' else settings.LABEL_LEN
    PRED_LEN = settings.PRED_LEN
    T = settings.T # data sampling interval

    # load model
    exp_name = f'{data_name}_{model_name}'
    save = f'checkpoints/{exp_name}_model.pt'
    with open(f'{save}', 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    # get data loader
    if model_name == 'nn':
        _, _, data_loader = get_data(data_name = data_name, label_len = SEQ_LEN)
    else:
        _, _, data_loader = get_data(data_name= data_name)

    # evaluate
    total_loss = []
    with torch.no_grad():
        for i, item in enumerate(data_loader):
            if data_name == 'SH_shift':
                enc_inp = item['his'][:,:, [0,1,4]].float().to(device)
            else:
                enc_inp = item['his'].float().to(device)

            batch_y = item['svSpd'].float()
            y_label = batch_y[:,-PRED_LEN:,:].to(device)
            batch_y_mark = item['lvSpd'].float().to(device)

            # decoder input
            if data_name == 'SH_shift':
                dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() 
            else:
                dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() + \
                        batch_y[:,:LABEL_LEN,:].mean(axis = 1, keepdim=True)
            dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
            dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
            
            # encoder - decoder
            if model_name == 'nn':
                out = model(dec_inp)
            else:
                out = model(enc_inp, dec_inp)
            
            lvSpd, spacing = item['lvSpd'][:, LABEL_LEN:,:].float().to(device), item['spacing'].float().to(device)
            relSpd_ = (lvSpd - out).squeeze()
            spacing_ = torch.cumsum(T*(relSpd_[:,:-1] + relSpd_[:,1:])/2, dim = -1) + item['s0'].float().to(device)
            loss = criterion(out, y_label) + criterion(spacing_, spacing)
            total_loss.append(loss.item())

    return np.mean(total_loss)

def main():
    model_names = [ 'transfollower', 'lstm', 'nn']
    for model_name in model_names:
        data_names = ['NGSIM', 'SH']
        if model_name == 'transfollower':
            data_names.append('SH_shift')

        for data_name in data_names:
            test_error = val(model_name, data_name)
            print(model_name, data_name, test_error)
    
if __name__ == '__main__':
    main()


