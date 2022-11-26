
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

class CarFolData(torch.utils.data.Dataset):
    """
    Dataset class for TransFollower and some baseline models.
    """
    def __init__(self, config, root = '/media/meixin/TransFollower/data/raw_data', split = 'train_SH_sample'):
        self.seq_len = config.SEQ_LEN
        self.max_len = config.MAX_LEN
        self.pred_len = self.max_len - self.seq_len
        self.label_len = config.LABEL_LEN
        self.data = np.load(os.path.join(root, split + '.npy'), allow_pickle = True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        event = self.data[idx]
        ret = dict()
        # ret['all'] = event
        ret['his'] = torch.from_numpy(event[:self.seq_len])
        ret['lvSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:self.max_len, 3])).unsqueeze(-1) 
        ret['svSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:self.max_len, 1])).unsqueeze(-1) 
        ret['spacing'] = torch.from_numpy(np.array(event[self.seq_len+1:self.max_len, 0])) 
        ret['s0'] = torch.from_numpy(np.array(event[self.seq_len, 0])).unsqueeze(-1)
        ret['lvSpdShift'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:self.max_len, -1])).unsqueeze(-1) 
        return ret 

class CarFolDataStar(torch.utils.data.Dataset):
    """
    Dataset for graph related models, e.g., STAR baseline. 
    """
    def __init__(self, config, root = '/media/meixin/TransFollower/data/raw_data', split = 'train_SH_sample'):
        self.data = np.load(os.path.join(root, split + '.npy'), allow_pickle = True)
        self.Ts = config.T
        self.MAX_LEN = config.MAX_LEN
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        event = self.data[idx][:self.MAX_LEN]
        lvSpd = event[:, 3]
        svSpd = event[:, 1]
        spacing = event[:, 0]
        
        svPos = np.cumsum((svSpd[1:] + svSpd[:-1])*0.5*self.Ts)
        svPos = np.append([0], svPos)
        lvPos = svPos + spacing

        lvFeat = np.concatenate([lvPos[:, None], lvSpd[:, None]], axis = -1) # T * d
        svFeat = np.concatenate([svPos[:, None], svSpd[:, None]], axis = -1)
        agents_feat = np.concatenate([lvFeat[None, :], svFeat[None,:]], axis = 0) # num_agents  * T * d
        agents_feat = torch.from_numpy(agents_feat).swapaxes(0, 1)
    
        return agents_feat

def get_data(config, data_name = 'SH', sample = False, shuffle = True):
    if not sample:
        train_dataset = CarFolData(split = f'train_{data_name}', config = config)
        val_dataset = CarFolData(split = f'val_{data_name}', config = config)
        test_dataset = CarFolData(split = f'test_{data_name}', config = config)
    else:
        train_dataset = CarFolData(split = f'train_{data_name}_sample', config = config)
        val_dataset = CarFolData(split = f'val_{data_name}_sample', config = config)
        test_dataset = CarFolData(split = f'test_{data_name}_sample', config = config)

    drop_last = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)

    return train_loader, val_loader, test_loader

def get_data_star(config, data_name = 'SH', sample = False, shuffle = True):
    if not sample:
        train_dataset = CarFolDataStar(split = f'train_{data_name}', config = config)
        val_dataset = CarFolDataStar(split = f'val_{data_name}', config = config)
        test_dataset = CarFolDataStar(split = f'test_{data_name}', config = config)
    else:
        train_dataset = CarFolDataStar(split = f'train_{data_name}_sample', config = config)
        val_dataset = CarFolDataStar(split = f'val_{data_name}_sample', config = config)
        test_dataset = CarFolDataStar(split = f'test_{data_name}_sample', config = config)

    drop_last = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=1,
        drop_last=drop_last)

    return train_loader, val_loader, test_loader

# class CarFolData_pretrain(torch.utils.data.Dataset):
#     def __init__(self, root = '/media/meixin/transfollower/data', seq_len = 40, 
#                  label_len =10, split = 'train_SH_pretrain'):
#         self.seq_len = seq_len
#         self.pred_len = 150 - self.seq_len
#         self.label_len = label_len

#         self.data = np.load(os.path.join(root, split + '_X' + '.npy'), allow_pickle = True)
#         self.label = np.load(os.path.join(root, split + '_y' + '.npy'), allow_pickle = True)
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         event = self.data[idx]
#         ret = dict()
#         ret['his'] = torch.from_numpy(event[:self.seq_len, [0,1,3]])
#         ret['lvSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 3])).unsqueeze(-1) 
#         ret['svSpd'] = torch.from_numpy(np.array(event[self.seq_len-self.label_len:150, 1])).unsqueeze(-1) 
#         ret['spacing'] = torch.from_numpy(np.array(event[self.seq_len+1:150, 0])) 
#         ret['s0'] = torch.from_numpy(np.array(event[self.seq_len, 0])).unsqueeze(-1)
#         ret['label'] = torch.from_numpy(np.array(self.label[idx]))
#         return ret

# def get_data_pretrain(data_name = 'SH', seq_len = SEQ_LEN, label_len = LABEL_LEN):
   
#     train_dataset = CarFolData_pretrain(seq_len = seq_len, split = f'train_{data_name}_pretrain', label_len = label_len)
#     val_dataset = CarFolData_pretrain(seq_len = seq_len, split = f'val_{data_name}_pretrain', label_len = label_len)
#     # test_dataset = CarFolData_pretrain(seq_len = seq_len, split = f'test_{data_name}_pretrain', label_len = label_len)

#     print(f'{data_name}, total data samples for train, val')
#     print(len(train_dataset), len(val_dataset))

#     drop_last = False
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=1,
#         drop_last=drop_last)
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=1,
#         drop_last=drop_last)
#     # test_loader = DataLoader(
#     #     test_dataset,
#     #     batch_size=BATCH_SIZE,
#     #     shuffle=True,
#     #     num_workers=1,
#     #     drop_last=drop_last)

#     print(f'Total batches for train, val with batch size {BATCH_SIZE}')
#     print(len(train_loader), len(val_loader))
#     return train_loader, val_loader

if __name__ == "__main__":
    dataset = CarFolDataStar()
    temp = dataset[0]
    train_loader, _, _ = get_data_star(sample = True)
    for item in train_loader:
        temp2 = item