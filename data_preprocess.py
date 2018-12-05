# -*- coding: utf-8 -*-
import os
import pandas as pd
import time
import config as cfg
from sklearn.preprocessing import LabelEncoder


def fillna(data, numeric_fts, string_fts):
    # 填充缺失值
    for ft in numeric_fts:
        data[ft].fillna(-1, inplace=True)

    for ft in string_fts:
        data[ft].fillna('-1', inplace=True)

    return data


def load_data():
    if not os.path.exists(cfg.op_train_sorted_file) \
            or not os.path.exists(cfg.tran_train_sorted_file) \
            or not os.path.exists(cfg.tag_train_sorted_file):
        print('[info]:start read from new train data...')
        is_preprocessed = False
        print('[info]:start read from op_train...')
        op_train = pd.read_csv(cfg.op_train_new_file)
        print('[info]:start read from tran_train...')
        tran_train = pd.read_csv(cfg.tran_train_new_file)
        print('[info]:start read from tag_train...')
        tag_train = pd.read_csv(cfg.tag_train_new_file)
    else:
        print('[info]:start read from sorted data...')
        is_preprocessed = True
        print('[info]:start read from op_train...')
        op_train = pd.read_csv(cfg.op_train_sorted_file).drop('Unnamed: 0', axis=1)
        print('[info]:start read from tran_train...')
        tran_train = pd.read_csv(cfg.tran_train_sorted_file).drop('Unnamed: 0', axis=1)
        print('[info]:start read from tag_train...')
        tag_train = pd.read_csv(cfg.tag_train_sorted_file).drop('Unnamed: 0', axis=1)

    if not is_preprocessed:
        preprocess(op_train, tran_train, tag_train)

    return op_train, tran_train, tag_train


def preprocess(op_train, tran_train, tag_train):
    # 处理时间字符串
    op_train['time'] = op_train['day'].apply(lambda x: "2018-08-%02d" % x) + ' ' + op_train['time']
    op_train['timestamp'] = op_train['time'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))

    tran_train['time'] = tran_train['day'].apply(lambda x: "2018-08-%02d" % x) + ' ' + tran_train['time']
    tran_train['timestamp'] = tran_train['time'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))

    # 根据UID进行排序 再根据timestamp进行排序
    op_train = op_train.sort_values(by=['UID', 'timestamp'], ascending=True).reset_index(drop=True)
    tran_train = tran_train.sort_values(by=['UID', 'timestamp'], ascending=True).reset_index(drop=True)
    tag_train = tag_train.sort_values(by=['UID'], ascending=True).reset_index(drop=True)

    op_train.to_csv(cfg.data_path + cfg.op_train_sorted_fn)
    tran_train.to_csv(cfg.data_path + cfg.tran_train_sorted_fn)
    tag_train.to_csv(cfg.data_path + cfg.tag_train_sorted_fn)
    return op_train, tran_train, tag_train


def drop_dup(op_train, tran_train, tag_train):
    # 去重
    op_train.drop_duplicates(inplace=True)
    tran_train.drop_duplicates(inplace=True)
    tag_train.drop_duplicates(inplace=True)
    return op_train, tran_train, tag_train
    # # 分组
    # op_train_gb = op_train.groupby('UID', as_index=False)
    # tran_train_gb = tran_train.groupby('UID', as_index=False)
    # #  获取op和tran各自的uid
    # op_train_uids = [uid for uid, item in op_train_gb.groups.items()]
    # tran_train_uids = [uid for uid, item in tran_train_gb.groups.items()]
    # # 获取op和tran各自的tag
    # op_tag = tag_train[tag_train['UID'].isin(op_train['UID'])]
    # tran_tag = tag_train[tag_train['UID'].isin(tran_train['UID'])]


def get_nan_counts(gb_count, ft_columns):
    hasnans_features_cnts = []
    for ft in ft_columns:
        cnts = gb_count[ft].value_counts()
        value = cnts[cnts.index == 0].values
        if len(value):
            hasnans_features_cnts.append((cnts.name, value[0]))
    return hasnans_features_cnts


def find_invalid_feature(gb_count, ft_columns):
    invalid_features = []
    for ft in ft_columns:
        cnts = gb_count[ft].value_counts()
        # 寻找值为0的统计数
        value = cnts[cnts.index == 0].values
        if len(value):
            if value[0] / gb_count.shape[0] > 0.5:
                print(cnts.name, value[0] / gb_count.shape[0])
                invalid_features.append(cnts.name)
    return invalid_features


def remove_list_item(src_l, rm_l):
    assert type(src_l) is list
    assert type(rm_l) is list

    for i in rm_l:
        if i in src_l:
            src_l.remove(i)
    return src_l


def process_data(train_data, train_columns, le_obj_fts, numtype_fts):
    print('[info]: start fill nans...')
    train_gb = None
    for ft in numtype_fts:
        train_data[ft].fillna(-1, inplace=True)
        train_gb = train_data.groupby('UID', as_index=False)

    # 假设用户的行为前后一致 填补缺失值
    # train_data = train_gb.ffill()
    # train_gb = train_data.groupby('UID', as_index=False)
    # train_data = train_gb.bfill()
    # train_gb = train_data.groupby('UID', as_index=False)

    # 在填补基础上计数，去除nan值占一半以上的值
    print('[info]: start remove invalid features...')
    invalid_features = find_invalid_feature(train_gb.count(), train_columns)
    train_columns = remove_list_item(train_columns, invalid_features)
    le_obj_fts = remove_list_item(le_obj_fts, invalid_features)
    train_data.drop(invalid_features, axis='columns', inplace=True)

    # 填补剩余的缺失值
    print('[info]: start handle left nans...')
    hasnans_features_cnts = get_nan_counts(train_gb.count(), train_columns)
    for ft_cnts in hasnans_features_cnts:
        if train_data[ft_cnts[0]].hasnans:
            train_data[ft_cnts[0]].fillna('-1', inplace=True)
    train_gb = train_data.groupby('UID', as_index=False)
    print('[info]: handle nans finished.')

    # 对非数值型标签进行编码
    print('[info]: start label encoding...')
    le = LabelEncoder()
    for feature in le_obj_fts:
        try:
            print('[info]: %r label encoding...' % feature)
            train_data[feature] = le.fit_transform(train_data[feature])
        except TypeError as e:
            print(e)
    train_gb = train_data.groupby('UID', as_index=False)
    print('[info]: label encoding finished.')
    return train_data, train_gb, train_columns, le_obj_fts


def drop_invalid(train_data, invalid_features):
    train_data.drop(invalid_features, axis='columns', inplace=True)
    return train_data


def label_encode(train_data, le_obj_fts):
    print('[info]: start label encoding...')
    le = LabelEncoder()
    for feature in le_obj_fts:
        try:
            print('[info]: %r label encoding...' % feature)
            train_data[feature] = le.fit_transform(train_data[feature])
        except TypeError as e:
            print(e)
    print('[info]: label encoding finished.')
    return train_data


def main():

    op_train, tran_train, tag_train = load_data()
    op_columns = list(op_train.columns)
    tran_columns = list(tran_train.columns)

    cfg.op_le_obj_fts = remove_list_item(cfg.op_le_obj_fts, cfg.op_drop_fts)
    cfg.op_numtype_fts = remove_list_item(cfg.op_numtype_fts, cfg.op_drop_fts)
    cfg.tran_le_obj_fts = remove_list_item(cfg.tran_le_obj_fts, cfg.tran_drop_fts)
    cfg.tran_numtype_fts = remove_list_item(cfg.tran_numtype_fts, cfg.tran_drop_fts)

    # 去重
    print('drop_dup...')
    op_train, tran_train, tag_train = drop_dup(op_train, tran_train, tag_train)
    # 去无用的ft
    print('drop...')
    op_train.drop(cfg.op_drop_fts, axis='columns', inplace=True)
    tran_train.drop(cfg.tran_drop_fts, axis='columns', inplace=True)
    # 填补缺失值
    print('fillna...')
    op_train = fillna(op_train, cfg.op_numtype_fts, cfg.op_le_obj_fts)
    tran_train = fillna(tran_train, cfg.tran_numtype_fts, cfg.tran_le_obj_fts)
    # 编码
    print('label_encode...')
    op_train = label_encode(op_train, cfg.op_le_obj_fts)
    tran_train = label_encode(tran_train, cfg.tran_le_obj_fts)

    op_train.to_csv(cfg.data_path + 'op_train.csv')
    tran_train.to_csv(cfg.data_path + 'tran_train.csv')


if __name__ == '__main__':
    main()
