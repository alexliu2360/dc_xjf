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


def fetch_op_fts(op_train):
    op_train_gb = op_train.groupby('UID')
    day_cnts = op_train_gb['day'].count()
    top_appear_day = op_train_gb['day'].value_counts().sort_values()


def fetch_tran_fts(tran_train):
    tran_train


def fetch_features(op_train, tran_train):
    fetch_op_fts(op_train)
    fetch_tran_fts(tran_train)

def main():
    op_train, tran_train, tag_train = load_data()
    fetch_features(op_train, tran_train)

if __name__ == '__main__':
    main()
