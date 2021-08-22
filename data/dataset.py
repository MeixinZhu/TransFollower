
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

import sys
if not '..' in sys.path:
    sys.path += ['..']

from config import Settings
settings = Settings()

# parameters
SEQ_LEN = settings.SEQ_LEN
LABEL_LEN = settings.LABEL_LEN
PRED_LEN = settings.PRED_LEN
BATCH_SIZE = settings.BATCH_SIZE

class CarFolData(torch.utils.data.Dataset):
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
        # ret['all'] = event
        ret['his'] = torch.from_numpy(event[:self.seq_len])
        ret['lvSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 3])).unsqueeze(-1) 
        ret['svSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 1])).unsqueeze(-1) 
        ret['spacing'] = torch.from_numpy(np.array(event[self.seq_len+1:150, 0])) 
        ret['s0'] = torch.from_numpy(np.array(event[self.seq_len, 0])).unsqueeze(-1)
        # ret['lvSpdShift'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 3])).unsqueeze(-1) 
        return ret 

def get_data(data_name = 'SH', sample = False, seq_len = SEQ_LEN, label_len = LABEL_LEN, shuffle = True, batch_size = BATCH_SIZE):
    if not sample:
        train_dataset = CarFolData(seq_len = seq_len, split = f'train_{data_name}', label_len = label_len)
        val_dataset = CarFolData(seq_len = seq_len, split = f'val_{data_name}', label_len = label_len)
        test_dataset = CarFolData(seq_len = seq_len, split = f'test_{data_name}', label_len = label_len)
    else:
        train_dataset = CarFolData(seq_len = seq_len, split = f'train_{data_name}_sample', label_len = label_len)
        val_dataset = CarFolData(seq_len = seq_len, split = f'val_{data_name}_sample', label_len = label_len)
        test_dataset = CarFolData(seq_len = seq_len, split = f'test_{data_name}_sample', label_len = label_len)
    # print(f'{data_name}, total data samples for train, val, and test')
    # print(len(train_dataset), len(val_dataset), len(test_dataset))

    drop_last = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)

    # print(f'Total batches for train, val, and test with batch size {BATCH_SIZE}')
    # print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader

class CarFolData_pretrain(torch.utils.data.Dataset):
    def __init__(self, root = '/media/meixin/transfollower/data', seq_len = 40, 
                 label_len =10, split = 'train_SH_pretrain'):
        self.seq_len = seq_len
        self.pred_len = 150 - self.seq_len
        self.label_len = label_len

        self.data = np.load(os.path.join(root, split + '_X' + '.npy'), allow_pickle = True)
        self.label = np.load(os.path.join(root, split + '_y' + '.npy'), allow_pickle = True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        event = self.data[idx]
        ret = dict()
        ret['his'] = torch.from_numpy(event[:self.seq_len, [0,1,3]])
        ret['lvSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 3])).unsqueeze(-1) 
        ret['svSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 1])).unsqueeze(-1) 
        ret['spacing'] = torch.from_numpy(np.array(event[self.seq_len+1:150, 0])) 
        ret['s0'] = torch.from_numpy(np.array(event[self.seq_len, 0])).unsqueeze(-1)
        ret['label'] = torch.from_numpy(np.array(self.label[idx]))
        return ret

def get_data_pretrain(data_name = 'SH', seq_len = SEQ_LEN, label_len = LABEL_LEN):
   
    train_dataset = CarFolData_pretrain(seq_len = seq_len, split = f'train_{data_name}_pretrain', label_len = label_len)
    val_dataset = CarFolData_pretrain(seq_len = seq_len, split = f'val_{data_name}_pretrain', label_len = label_len)
    # test_dataset = CarFolData_pretrain(seq_len = seq_len, split = f'test_{data_name}_pretrain', label_len = label_len)

    print(f'{data_name}, total data samples for train, val')
    print(len(train_dataset), len(val_dataset))

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
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=1,
    #     drop_last=drop_last)

    print(f'Total batches for train, val with batch size {BATCH_SIZE}')
    print(len(train_loader), len(val_loader))
    return train_loader, val_loader