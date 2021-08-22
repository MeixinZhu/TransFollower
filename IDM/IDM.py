import numpy as np

def IDM(para, spacing, svSpd, relSpd):
    # para is a vector containing IDM's parameters
    desiredSpd = para[0] #in m/s
    desiredTimeHdw = para[1] #in seconds
    maxAcc = para[2] # m/s^2
    comfortAcc = para[3] # m/s^2
    beta = para[4]
    jamSpace = para[5] #in meters
    
    desiredSpacing = jamSpace + max(0, desiredTimeHdw*svSpd-svSpd*relSpd/(2*np.sqrt(maxAcc*comfortAcc)))
    acc = maxAcc*(1-(svSpd/desiredSpd)**beta-(desiredSpacing/spacing)**2)
    
    return acc

def IDM_sim(para, event, context = 40):
    # para is a vector containing IDM parameters
    # event is the car following event with columns [spacing, svSpd, relSpd, lvSpd]
    # context is the amount of data used as historical data
    
    T = 0.1 # data sampling interval
    MAX_LEN = 150
    
    svSpd_sim = []
    spacing_sim = []
    
    spacing, svSpd, relSpd = event[context-1][:-1]
    
    lvSpd = event[:, -1]
    
    for i in range(context, MAX_LEN):
        acc = IDM(para, spacing, svSpd, relSpd)

        svSpd_ = max(0.001, svSpd + acc*T) # next step svSpd
        relSpd_ = lvSpd[i] - svSpd_
        
        spacing_ = spacing + T*(relSpd_ + relSpd)/2 
        
        svSpd = svSpd_
        relSpd = relSpd_
        spacing = spacing_
        
        svSpd_sim.append(svSpd)
        spacing_sim.append(spacing)
        
    svSpd_obs = event[context:MAX_LEN, 1]
    spacing_obs = event[context:MAX_LEN:, 0]
    lvSpd_obs = event[context:MAX_LEN, -1]
    
    return svSpd_obs, spacing_obs, lvSpd_obs, svSpd_sim, spacing_sim

