import numpy as np
import os
from config import Settings, HighDSettings

# load data
DATASET = 'highD' 
# save the calibrated paramters
MODEL = 'IDM'
exp_name = f'{DATASET}_{MODEL}'
save = f'checkpoints/{exp_name}_model.npy'
COLLISION_PENALTY = 100

if DATASET == 'highD':
    settings = HighDSettings()
else:
    settings = Settings()

SEQ_LEN = 0 #settings.SEQ_LEN

def IDM(para, spacing, svSpd, relSpd):
    """
    Funciton that takes IDM paramters and car-following states as inputs, and output
    the acceleration for the following vehicle.

    :param para: a vector containing IDM's parameters. 
        E.g., para = np.array([32.0489077 ,  0.74084102,  1.18623382,  0.87773747,  1.,2.95210611])
    :param spacing: scaler, gap between two vehicles [m].
    :param svSpd: speed of the following vehicle [m/s].
    :param relSpd: lead vehicle speed - following vehicle speed [m/s].
    :return: acceleration of the following vehicle in next step [m/s^2]. 
    """
    desiredSpd = para[0] #in m/s
    desiredTimeHdw = para[1] #in seconds
    maxAcc = para[2] # m/s^2
    comfortAcc = para[3] # m/s^2
    beta = para[4]
    jamSpace = para[5] #in meters

    desiredSpacing = jamSpace + max(0, desiredTimeHdw*svSpd-svSpd*relSpd/(2*np.sqrt(maxAcc*comfortAcc)))
    acc = maxAcc*(1-(svSpd/desiredSpd)**beta-(desiredSpacing/spacing)**2)
    
    return acc

def simulate_car_fol(model_fun, lvSpd, init_s, init_svSpd, para):
    """
    Simulate a car following event based on a car-following model.

    :param model_fun:
    """
    T = settings.T # data sampling interval

    svSpd_sim = []
    spacing_sim = []
    spacing, svSpd, relSpd = init_s, init_svSpd, lvSpd[0] - init_svSpd

    svSpd_sim.append(svSpd)
    spacing_sim.append(spacing)

    collision_cost = 0

    for i in range(1, len(lvSpd)):
        # calcualate next_step acceleration using IDM model
        acc = model_fun(para, spacing, svSpd, relSpd)

        # state update based on Newton's motion law
        svSpd_ = max(0.001, svSpd + acc*T) # next step svSpd
        relSpd_ = lvSpd[i] - svSpd_
        spacing_ = spacing + T*(relSpd_ + relSpd)/2 
        
        # update state variables 
        svSpd = svSpd_
        relSpd = relSpd_
        spacing = spacing_

        if spacing < 0:
            collision_cost = (len(lvSpd) - i)*COLLISION_PENALTY
            break
        
        # store simulation results
        svSpd_sim.append(svSpd)
        spacing_sim.append(spacing)
    
    return np.asarray(svSpd_sim), np.asarray(spacing_sim), collision_cost

"""## Model calibration based on the simulated trajectory"""
def evaluate(data, para):
    #[space, svSpd, relSpd, lvSpd]
    total_error = 0
    for event in data:
        space, svSpd, lvSpd = event[:, 0], event[:, 1], event[:, -1]
        init_s, init_svSpd = event[0, 0], event[0, 1]
        svSpd_sim, spacing_sim, collision_cost = simulate_car_fol(
            IDM, lvSpd=lvSpd, init_s=init_s, init_svSpd=init_svSpd, para=para)
        error = np.mean((svSpd_sim[SEQ_LEN:] - svSpd[SEQ_LEN:len(svSpd_sim)])**2) + \
             np.mean((spacing_sim[SEQ_LEN:] - space[SEQ_LEN:len(spacing_sim)])**2) + collision_cost
        total_error += error
    return total_error/len(data) # avg of MSE speed and space

if __name__ == "__main__":
    root = '/media/meixin/TransFollower/data/raw_data'
    test_data = np.load(os.path.join(root, f'test_{DATASET}' + '.npy'), allow_pickle = True)

    para = np.load(save)
    error = evaluate(test_data, para)
    
    print(f'{MODEL}, {DATASET}, testing error: {error}')



