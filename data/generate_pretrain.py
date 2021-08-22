import numpy as np
import os
from tqdm import tqdm

root = '/media/meixin/transfollower/data'

split = 'train_SH'

# [space, svSpd, relSpd, lvSpd] 
data = np.load(os.path.join(root, split + '.npy'), allow_pickle = True)

N = len(data)
y = []
X = []

MAX_LEN = 150

for i in tqdm(range(N)):
    event = data[i][:MAX_LEN]
    X.append(event)
    y.append(1)
    
    # randomly select another event's data 
    j = np.random.randint(0, N)
    fake_lvSpd = data[j][:MAX_LEN, -1]
    X.append(np.hstack((event[:,:-1], fake_lvSpd[:,None] )))
    y.append(0)
    
np.save(os.path.join(root, split + '_pretrain_X' + '.npy'), X)
np.save(os.path.join(root, split + '_pretrain_y' + '.npy'), y)