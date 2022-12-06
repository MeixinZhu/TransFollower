import torch
from torch import nn
import numpy as np

from data.dataset import get_data, get_data_star
from model.model import Transfollower, lstm_model, nn_model

from config import Settings, HighDSettings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

def val(model_name, data_name):
    if data_name == 'highD':
        settings = HighDSettings()
    else:
        settings = Settings()

    # parameters
    SEQ_LEN = settings.SEQ_LEN
    if model_name == 'nn':
        settings.LABEL_LEN = SEQ_LEN
    LABEL_LEN = settings.LABEL_LEN
    PRED_LEN = settings.PRED_LEN
    Ts = settings.T # data sampling interval

    # load model
    exp_name = f'{data_name}_{model_name}'
    save = f'checkpoints/{exp_name}_model.pt'
    with open(f'{save}', 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    if model_name in ['trajectron', 'star']:
        _, _, data_loader = get_data_star(data_name = data_name, config = settings)
    else:
        _, _, data_loader = get_data(data_name = data_name, config = settings)

    # evaluate
    total_loss = []
    with torch.no_grad():
        for i, item in enumerate(data_loader):
            if model_name in ['trajectron', 'star']:
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
            else:
                enc_inp = item['his'].float().to(device)
                batch_y = item['svSpd'].float()
                y_label = batch_y[:,-PRED_LEN:,:].to(device)

                # decoder input
                batch_y_mark = item['lvSpd'].float().to(device)
                dec_inp = torch.zeros([batch_y.shape[0], PRED_LEN, batch_y.shape[-1]]).float() + \
                        batch_y[:,:LABEL_LEN,:].mean(axis = 1, keepdim=True)
                dec_inp = torch.cat([batch_y[:,:LABEL_LEN,:], dec_inp], dim=1).float().to(device)
                dec_inp = torch.cat([dec_inp, batch_y_mark], axis = -1) # adding lv speed
                
                # encoder - decoder
                if model_name == 'nn':
                    out = model(dec_inp)
                elif model_name == 'transfollower':
                    out = model(enc_inp, dec_inp)[0]
                else:
                    out = model(enc_inp, dec_inp)
                
                lvSpd, spacing = item['lvSpd'][:, LABEL_LEN:,:].float().to(device), item['spacing'].float().to(device)
                relSpd_ = (lvSpd - out).squeeze()
                spacing_ = torch.cumsum(Ts*(relSpd_[:,:-1] + relSpd_[:,1:])/2, dim = -1) + item['s0'].float().to(device)
                loss = criterion(out, y_label) + criterion(spacing_, spacing)

            total_loss.append(loss.item())

    return np.mean(total_loss)

def main():
    model_names = ['transfollower', 'lstm', 'nn', 'trajectron', 'star']
    for model_name in model_names:
        data_names = ['SH', 'highD']

        for data_name in data_names:
            test_error = val(model_name, data_name)
            print(model_name, data_name, test_error)
    
if __name__ == '__main__':
    main()

