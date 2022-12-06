# TransFollower: Long-Sequence Car-Following Trajectory Prediction through Transformer
Source code for the following paper:

Zhu, Meixin, et al. "TransFollower: Long-Sequence Car-Following Trajectory Prediction through Transformer." arXiv preprint arXiv:2202.03183 (2022).

## Description
To model the long-term dependency of future actions on historical driving situations, we developed a long-sequence car-following trajectory prediction model based on the attention-based Transformer model. The model follows a general format of encoder-decoder architecture. The encoder takes historical speed and spacing data as inputs and forms a mixed representation of historical driving context using multi-head self-attention. The decoder takes the future LV speed profile as input and outputs the predicted future FV speed profile in a generative way (instead of an auto-regressive way, avoiding compounding errors). Through cross-attention between encoder and decoder, the decoder learns to build a connection between historical driving and future LV speed, based on which a prediction of future FV speed can be obtained. We train and test our model with real-world car-following events extracted from the Shanghai Naturalistic Driving Study (SH-NDS) and HighD datasets. Results show that the model outperforms the traditional intelligent driver model (IDM), IDM with memory effects, a fully connected neural network model, a long short-term memory (LSTM) based model, and customized state-of-the-art trajectory prediction models in terms of long-sequence trajectory prediction accuracy. We also visualized the self-attention and cross-attention heatmaps to explain how the model derives its predictions. 

## Set Up Environment
`pip install -r requirements.txt`

## Data
The model has been tested with SH-NDS, highD, and NGSIM. Extracted car-following events are stored in `data/raw_data` folder. For privacy issues, SH-NDS related data is not published here, please contact the authors for access. You could use highD data for experiments first. 

## Train 
Train transfollower, lstm, and nn models:  
`python train.py` 

Train Trajectron++ model:   
`python train_trajectron.py`   

Train STAR model:   
`python train_star.py`

Train IDM/IDMM models:     
`python IDM_calibrate.py` or `python IDMM_calibrate.py`

## Test
Test of all data-driven models using model checkpoints stored in `checkpoint/`:   
`python test.py`

Test IDM/IDMM models:     
`python IDM_test.py` or `python IDMM_test.py`

## Pretrained Models
Pretrained models are stored in `checkpoints/` folder. 

## Reference
Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus   
STAR: https://github.com/Majiker/STAR   

## Contact
meixin@ust.hk