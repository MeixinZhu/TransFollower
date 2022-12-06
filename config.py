class Settings:
    SEQ_LEN = 40
    LABEL_LEN = 10
    MAX_LEN = 150 # max number of steps for one event
    PRED_LEN = MAX_LEN - SEQ_LEN
    BATCH_SIZE = 256
    lr = 1e-3
    T = 0.1 # data sampling interval
    N_EPOCHES = 1000

# for highD
class HighDSettings:
    SEQ_LEN = 50
    LABEL_LEN = 12
    MAX_LEN = 187 # max number of steps for one event
    PRED_LEN = MAX_LEN - SEQ_LEN
    BATCH_SIZE = 256
    lr = 1e-3
    T = 0.08 # data sampling interval
    N_EPOCHES = 1000