# -*- coding: utf-8 -*-
import os

base_path = os.path.abspath(os.path.dirname(__file__))

data_path = base_path + '/../data/'
origin_path = data_path + 'origin_data/'
fts_path = data_path + 'fts/'

op_train_fts_fn = 'op_fts.csv'
tran_train_fts_fn = 'tran_fts.csv'
op_train_fts_round1_fn = 'op_round1_fts.csv'
tran_train_fts_round1_fn = 'tran_round1_fts.csv'
tag_train_fts_fn = 'tag_fts.csv'
round1_fts_fn = 'round1_fts.csv'

op_train_fts_file = fts_path + op_train_fts_fn
tran_train_fts_file = fts_path + tran_train_fts_fn
op_train_fts_round1_file = fts_path + op_train_fts_round1_fn
tran_train_fts_round1_file = fts_path + tran_train_fts_round1_fn
round1_fts_file = fts_path + round1_fts_fn


op_train_new_fn = 'operation_train_new.csv'
tran_train_new_fn = 'transaction_train_new.csv'
tag_train_new_fn = 'tag_train_new.csv'
op_train_round1_fn = 'operation_round1_new.csv'
tran_train_round1_fn = 'transaction_round1_new.csv'

op_train_sorted_fn = 'op_train_sorted.csv'
tran_train_sorted_fn = 'tran_train_sorted.csv'
tag_train_sorted_fn = 'tag_train_sorted.csv'
op_train_round1_sorted_fn = 'operation_round1_sorted.csv'
tran_train_round1_sorted_fn = 'transaction_round1_sorted.csv'

op_origin_fn = 'op_origin.csv'
tran_origin_fn = 'tran_origin.csv'
op_origin_round1_fn = 'op_origin_round1.csv'
tran_origin_round1_fn = 'tran_origin_round1.csv'


tag_train_fts_file = fts_path + tag_train_fts_fn
op_train_new_file = data_path + op_train_new_fn
tran_train_new_file = data_path + tran_train_new_fn
tag_train_new_file = data_path + tag_train_new_fn

op_train_round1_file = data_path + op_train_round1_fn
tran_train_round1_file = data_path + tran_train_round1_fn


op_train_sorted_file = data_path + op_train_sorted_fn
tran_train_sorted_file = data_path + tran_train_sorted_fn
tag_train_sorted_file = data_path + tag_train_sorted_fn
op_train_round1_sorted_file = data_path + op_train_round1_sorted_fn
tran_train_round1_sorted_file = data_path + tran_train_round1_sorted_fn

op_origin_train_file = origin_path + op_origin_fn
tran_origin_train_file = origin_path + tran_origin_fn
op_origin_round1_file = origin_path + op_origin_round1_fn
tran_origin_round1_file = origin_path + tran_origin_round1_fn

op_obj_fts = ['mode', 'os', 'version',
                 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3',
                 'mac1', 'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']
op_le_obj_fts = ['mode', 'os', 'version',
                 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3',
                 'mac1', 'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub', 'device_code', 'ipsub', 'ip']
op_numtype_fts = ['success']
tran_obj_fts = ['amt_src1', 'merchant',
                   'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
                   'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
                   'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
                   'market_code', 'ip1_sub']
tran_le_obj_fts = ['amt_src1', 'merchant',
                   'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
                   'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
                   'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
                   'market_code', 'ip1_sub', 'device_code']
tran_numtype_fts = ['channel', 'trans_amt', 'trans_type2', 'market_type']

op_drop_fts = ['mac2', 'geo_code', 'wifi', 'os', 'version']
tran_drop_fts = ['acc_id2', 'acc_id3', 'market_type', 'code2', 'geo_code']

tran_columns = ['UID', 'channel', 'day', 'time', 'trans_amt', 'amt_src1', 'merchant',
       'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
       'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
       'bal', 'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
       'market_code', 'market_type', 'ip1_sub', 'timestamp']

