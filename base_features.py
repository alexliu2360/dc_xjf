# coding='utf8'
import os
import config as cfg
import pandas as pd

def load_data():
    print('[info]:start read from op_train...')
    op_train = pd.read_csv(cfg.op_origin_file).drop('Unnamed: 0', axis=1)
    print('[info]:start read from tran_train...')
    tran_train = pd.read_csv(cfg.tran_origin_file).drop('Unnamed: 0', axis=1)
    print('[info]:start read from tag_train...')
    tag_train = pd.read_csv(cfg.tag_train_sorted_file).drop('Unnamed: 0', axis=1)
    return op_train, tran_train, tag_train


