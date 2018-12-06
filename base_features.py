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
    # op_train_gb = op_train.groupby('UID')
    # day_cnts = op_train_gb['day'].count()
    # top_appear_day = op_train_gb['day'].value_counts().sort_values()
    uids = []
    features = []
    for index, row in op_train.iterrows():
        if index / 100 == 20:
            break
        if index % 100 == 0:
            print(index)
        if row['UID'] in uids:
            continue
        else:
            uids.append(row['UID'])
        feature = {}
        feature['UID'] = row['UID']
        tmp = op_train[op_train['UID'] == row['UID']]
        feature['day_cnts'] = tmp['day'].count()
        feature['top_appear_day'] = tmp['day'].value_counts()
        features.append(feature)
    features = pd.DataFrame(features)
    return features


def fetch_tran_fts(tran_train):
    # tran_train
    pass


def fetch_features(op_train, tran_train):
    fetch_op_fts(op_train)
    fetch_tran_fts(tran_train)


def main():
    op_train, tran_train, tag_train = load_data()
    fetch_features(op_train, tran_train)


if __name__ == '__main__':
    main()
