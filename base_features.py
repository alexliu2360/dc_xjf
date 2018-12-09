# coding='utf8'
import os
import config as cfg
import pandas as pd
import numpy as np


def load_new_data():
    print('[info]:start read from op_train...')
    op_train = pd.read_csv(cfg.op_origin_train_file).drop('Unnamed: 0', axis=1)
    print('[info]:start read from tran_train...')
    tran_train = pd.read_csv(cfg.tran_origin_train_file).drop('Unnamed: 0', axis=1)
    print('[info]:start read from tag_train...')
    tag_train = pd.read_csv(cfg.tag_train_sorted_file).drop('Unnamed: 0', axis=1)
    return op_train, tran_train, tag_train


def load_round1_data():
    print('[info]:start read from op_rd1...')
    op_rd1 = pd.read_csv(cfg.op_origin_round1_file).drop('Unnamed: 0', axis=1)
    print('[info]:start read from tran_rd1...')
    tran_rd1 = pd.read_csv(cfg.tran_origin_round1_file).drop('Unnamed: 0', axis=1)
    return op_rd1, tran_rd1


def load_fts():
    op_fts = pd.read_csv(cfg.op_train_fts_file)
    tran_fts = pd.read_csv(cfg.tran_train_fts_file)
    return op_fts, tran_fts


def oneday_cnt(tmp, gb_str, ft_str):
    gb = tmp.groupby(gb_str)
    top_oneday = []
    for gb_key in gb.indices.keys():
        cnt = 0
        sub_gb = gb.get_group(gb_key)
        value_counts = sub_gb[ft_str].value_counts()
        if not value_counts.empty:
            top_value = value_counts.sort_values(ascending=False).values[0]
            top_mode = value_counts.sort_values(ascending=False).index[0]
            top_oneday.append([gb_key, top_value, top_mode])
    if not len(top_oneday):
        top_oneday = [np.nan, np.nan, np.nan]
    return top_oneday


def topcnts_oneday(tmp, sub_str):
    top_oneday = oneday_cnt(tmp, 'day', sub_str)
    if pd.Series(top_oneday).hasnans:
        return np.nan, np.nan, np.nan
    top_idx = 0
    top_cnt = top_oneday[0][1]
    for item in top_oneday:
        if top_cnt < item[1]:
            top_cnt = item[1]
            top_idx += 1
    top_day = top_oneday[top_idx][0]
    top_value = top_oneday[top_idx][1]
    top_mode = top_oneday[top_idx][2]
    return top_day, top_value, top_mode


def op_times_per2min(tmp):
    day_gb = tmp.groupby('day')
    min2 = 120
    over2min_rec = []
    for gb_key in day_gb.indices.keys():
        over2min_cnt = 0
        sgb = day_gb.get_group(gb_key)
        timestamp = list(sgb['timestamp'].get_values())
        index, idx_max = 0, len(timestamp) - 1
        now_t = timestamp[index]
        for _ in timestamp:
            if timestamp[index] - now_t > min2:
                now_t = timestamp[index]
                over2min_cnt += 1
            index += 1
        over2min_rec.append(over2min_cnt)
    return max(over2min_rec)


def suc_rate(tmp):
    if not tmp['success'].value_counts().empty:
        suc_cnt = 0
        for item in tmp['success'].value_counts().items():
            if item[0] == 1:
                suc_cnt = item[1]
                break
        suc_all = tmp['success'].count()
        return suc_cnt / suc_all
    else:
        return np.nan


def fbfill_series(series):
    new_s = series.copy()
    new_s.fillna(method='bfill', inplace=True)
    new_s.fillna(method='ffill', inplace=True)
    return new_s


def ft_change_cnt(tmp, ft_str):
    assert ft_str in tmp.keys()

    # 排除某一个ft下全部是NaN的情况
    if len(tmp[ft_str][tmp[ft_str].notna()]):
        series = fbfill_series(tmp[ft_str])
        last_dc = series.get_values()[0]
        dc_cnt = 0
        for dc in series.get_values():
            if last_dc != dc:
                last_dc = dc
                dc_cnt += 1
        return dc_cnt
    else:
        return np.nan


def get_change_frq(tmp, ft_str):
    frq = ft_change_cnt(tmp, ft_str) / tmp['day'].count()
    if frq is not np.nan:
        return float('%.2f' % frq)
    else:
        return np.nan


def ip_change_oneday_top(tmp, ip_ft_str):
    day_gb = tmp.groupby('day')
    ip_rec = []
    for gb_key in day_gb.indices.keys():
        sgb = day_gb.get_group(gb_key)
        series = fbfill_series(sgb[ip_ft_str])
        last_ip = series.get_values()[0]
        ip_cnt = 0
        if np.isnan(last_ip):
            ip_rec.append(np.nan)
            continue
        for ip in series.get_values():
            if last_ip != ip:
                last_ip = ip
                ip_cnt += 1
        ip_rec.append(ip_cnt)
    return max(ip_rec)


def top_type(tmp, ft_str):
    value_counts = tmp[ft_str].value_counts()
    if not value_counts.empty:
        return value_counts.sort_values(ascending=False).index[0]
    else:
        return np.nan


def top_value(tmp, ft_str):
    value_counts = tmp[ft_str].value_counts()
    if not value_counts.empty:
        return value_counts.sort_values(ascending=False).values[0]
    else:
        return np.nan


def get_frq(tmp, ft_str):
    return float('%.2f' % (top_value(tmp, ft_str) / tmp['day'].count()))


def top_type_in_diffUID_cnt(tmp, ft_str, type_dict):
    tp = top_type(tmp, ft_str)
    if tp and tp in type_dict.keys():
        return type_dict[tp]
    else:
        return np.nan


def fetch_op_fts(op_train):
    uids = []
    features = []
    op_mac1_dict = op_train['mac1'].value_counts().to_dict()
    op_ipsub_dict = op_train['ipsub'].value_counts().to_dict()
    for index, row in op_train.iterrows():
        if index % 10000 == 0:
            print(index)
        if row['UID'] in uids:
            continue
        else:
            uids.append(row['UID'])
        op_feature = {}
        op_feature['UID'] = row['UID']
        tmp = op_train[op_train['UID'] == row['UID']]
        op_feature['UID'] = tmp['UID'].values[0]
        op_feature['day_cnts'] = tmp['day'].count()
        op_feature['op_top_appear_day'] = top_type(tmp, 'day')
        op_feature['op_top_appear_day_cnt'] = top_value(tmp, 'day')
        op_feature['op_times_per2min'] = op_times_per2min(tmp)
        op_feature['mode_top_day_oneday'] = topcnts_oneday(tmp, 'mode')[0]  # 一天中某一操作类型次数最多的那一天
        op_feature['mode_top_cnt_oneday'] = topcnts_oneday(tmp, 'mode')[1]  # 一天中某一操作类型次数最多的次数
        op_feature['mode_top_type_oneday'] = topcnts_oneday(tmp, 'mode')[2]  # 一天中某一操作类型次数最多的类型
        op_feature['mode_cnt'] = top_value(tmp, 'mode')
        op_feature['mode_rank1'] = top_type(tmp, 'mode')
        op_feature['suc_rate'] = '%.2f' % (suc_rate(tmp))
        op_feature['device_code_frq'] = get_change_frq(tmp, 'device_code')
        op_feature['ip_change_frq'] = get_change_frq(tmp, 'ip')
        op_feature['ip_change_oneday_top'] = ip_change_oneday_top(tmp, 'ip')
        op_feature['ip_change_cnt'] = ft_change_cnt(tmp, 'ip')
        op_feature['top_mac1_in_diffUID_cnt'] = top_type_in_diffUID_cnt(tmp, 'mac1', op_mac1_dict)
        op_feature['top_ipsub_in_diffUID_cnt'] = top_type_in_diffUID_cnt(tmp, 'ipsub', op_ipsub_dict)

        features.append(op_feature)
    features = pd.DataFrame(features)
    features.to_csv(cfg.op_train_fts_file)
    return features


def fetch_tran_fts(tran_train):
    uids = []
    features = []
    tran_mac1_dict = tran_train['mac1'].value_counts().to_dict()
    tran_ip1sub_dict = tran_train['ip1_sub'].value_counts().to_dict()
    for index, row in tran_train.iterrows():
        if index % 10000 == 0:
            print(index)
        if row['UID'] in uids:
            continue
        else:
            uids.append(row['UID'])
        tmp = tran_train[tran_train['UID'] == row['UID']]
        tran_feature = {}
        tran_feature['UID'] = tmp['UID'].values[0]
        tran_feature['channel_top'] = top_type(tmp, 'channel')
        tran_feature['channel_top_frq'] = top_value(tmp, 'channel')
        tran_feature['tran_day_cnts'] = tmp['day'].count()
        tran_feature['tran_day_appear_top'] = top_type(tmp, 'day')
        tran_feature['tran_amt_frq'] = get_frq(tmp, 'trans_amt')
        tran_feature['tran_amt_top'] = top_type(tmp, 'trans_amt')
        tran_feature['tran_topcnts_oneday'] = topcnts_oneday(tmp, 'trans_amt')[1]
        tran_feature['tran_times_per2min'] = op_times_per2min(tmp)
        tran_feature['amt_src1_frq'] = get_change_frq(tmp, 'amt_src1')
        tran_feature['amt_src1_type_top'] = top_type(tmp, 'amt_src1')
        tran_feature['amt_src1_type_cnt'] = topcnts_oneday(tmp, 'amt_src1')[1]
        tran_feature['amt_src2_frq'] = get_change_frq(tmp, 'amt_src2')
        tran_feature['amt_src2_type_top'] = top_type(tmp, 'amt_src2')
        tran_feature['amt_src2_type_cnt'] = topcnts_oneday(tmp, 'amt_src2')[1]
        tran_feature['merchant_frq'] = get_change_frq(tmp, 'merchant')
        tran_feature['merchant_type_top'] = top_type(tmp, 'merchant')
        tran_feature['merchant_type_cnt'] = len(tmp['merchant'].value_counts())  # 商户标识类型总数
        tran_feature['code1_type_top'] = top_type(tmp, 'code1')
        tran_feature['code1_type_cnt'] = len(tmp['code1'].value_counts())  # 出现最多的商户子门店
        tran_feature['trans_type1_top_cnt'] = top_value(tmp, 'trans_type1')
        tran_feature['trans_type1_top_frq'] = get_frq(tmp, 'trans_type1')
        tran_feature['trans_type1_top'] = top_type(tmp, 'trans_type1')
        tran_feature['trans_type2_top_cnt'] = top_value(tmp, 'trans_type2')
        tran_feature['trans_type2_top_frq'] = get_frq(tmp, 'trans_type2')
        tran_feature['trans_type2_top'] = top_type(tmp, 'trans_type2')
        tran_feature['acc_id1_top_cnt'] = top_value(tmp, 'acc_id1')
        tran_feature['acc_id1_top_frq'] = get_frq(tmp, 'acc_id1')
        tran_feature['acc_id1_top'] = top_type(tmp, 'acc_id1')
        tran_feature['device_code_frq'] = get_change_frq(tmp, 'device_code')
        tran_feature['dev_name_frq'] = get_change_frq(tmp, 'device1')
        tran_feature['dev_type_frq'] = get_change_frq(tmp, 'device2')
        tran_feature['ip_change_oneday_top'] = ip_change_oneday_top(tmp, 'ip1')
        tran_feature['ip_change_frq'] = get_change_frq(tmp, 'ip1')  # ip变化次数
        tran_feature['ip_change_times'] = ft_change_cnt(tmp, 'ip1')  # ip变化次数
        tran_feature['top_mac1_in_diffUID_cnt'] = top_type_in_diffUID_cnt(tmp, 'mac1', tran_mac1_dict)
        tran_feature['top_ip1sub_in_diffUID_cnt'] = top_type_in_diffUID_cnt(tmp, 'ip1_sub', tran_ip1sub_dict)
        features.append(tran_feature)
    features = pd.DataFrame(features)
    features.to_csv(cfg.tran_train_fts_file)
    return features


def get_feature(op, trans, tag):
    tag = tag.merge(op, on='UID', how='left')
    tag = tag.merge(trans, on='UID', how='left')
    return tag


def merge_feature(op, tran):
    rd1 = op.merge(tran, on='UID', how='left')
    return rd1


def train_data():
    print('train data...')
    op_train, tran_train, tag_train = load_new_data()
    if not os.path.exists(cfg.op_train_fts_file):
        print('fetch_op_fts...')
        op_fts = fetch_op_fts(op_train)
    else:
        print('load op_fts...')
        op_fts = pd.read_csv(cfg.op_train_fts_file)

    if not os.path.exists(cfg.tran_train_fts_file):
        print('fetch_tran_fts...')
        tran_fts = fetch_tran_fts(tran_train)
    else:
        print('load tran_fts...')
        tran_fts = pd.read_csv(cfg.tran_train_fts_file)

    if not os.path.exists(cfg.tag_train_fts_file):
        print('get_feature...')
        tag = get_feature(op_fts, tran_fts, tag_train)
        tag.to_csv(cfg.tag_train_fts_file)


def round1_data():
    print('round1 data...')
    op_rd1, tran_rd1 = load_round1_data()
    if not os.path.exists(cfg.op_train_fts_round1_file):
        print('fetch_op_fts...')
        op_fts = fetch_op_fts(op_rd1)
    else:
        print('load op_fts...')
        op_fts = pd.read_csv(cfg.op_train_fts_round1_file)

    if not os.path.exists(cfg.tran_train_fts_round1_file):
        print('fetch_tran_fts...')
        tran_fts = fetch_tran_fts(tran_rd1)
    else:
        print('load tran_fts...')
        tran_fts = pd.read_csv(cfg.tran_train_fts_round1_file)

    if not os.path.exists(cfg.round1_fts_file):
        print('merge_feature...')
        round1_fts = merge_feature(op_fts, tran_fts)
        round1_fts.to_csv(cfg.round1_fts_file)


def main():
    train_data()
    round1_data()


if __name__ == '__main__':
    main()
