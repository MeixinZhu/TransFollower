class Settings:
    SEQ_LEN = 40
    LABEL_LEN = 10
    PRED_LEN = 150 - SEQ_LEN
    BATCH_SIZE = 512
    lr = 6e-5
    T = 0.1 # data sampling interval
    N_EPOCHES = 1000