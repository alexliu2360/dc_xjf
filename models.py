from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
import config as cfg
import numpy as np


def tpr_weight_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    return 'TC_AUC', 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3, True


def load_train_test_data():
    train_data = pd.read_csv(cfg.tag_train_fts_file)
    test_data = pd.read_csv(cfg.round1_fts_file)
    return train_data, test_data

def fillna(data):
    data.fillna(-1, inplace=True)
    return data


def LGB_test(train_x, train_y, test_x, test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
        learning_rate=0.01, min_child_weight=25, random_state=2018, n_jobs=50
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], early_stopping_rounds=100)
    feature_importances = sorted(zip(train_x.columns, clf.feature_importances_), key=lambda x: x[1])
    return clf.best_score_['valid_1']['binary_logloss'], feature_importances


def off_test_split(data, cate_col=None):
    y = data.pop('Tag')
    train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.15, random_state=2018)
    score = LGB_test(train_x, train_y, test_x, test_y)
    return score[1]


def LGB_predict(data, file):
    train = data[data['is_trade'] > -1]
    predict = data[data['is_trade'] == -2]
    res = predict[['instance_id']]
    train_y = train.pop('is_trade')
    train_x = train.drop(['day', 'instance_id'], axis=1)
    test_x = predict.drop(['day', 'instance_id', 'is_trade'], axis=1)
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
        learning_rate=0.01, min_child_weight=25, random_state=2018, n_jobs=50
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)])
    res['predicted_score'] = clf.predict_proba(test_x)[:, 1]
    testb = pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt', sep=' ')[['instance_id']]
    res = pd.merge(testb, res, on='instance_id', how='left')
    res[['instance_id', 'predicted_score']].to_csv('../submit/' + file + '.txt', sep=' ', index=False)


def add(f1, f2):
    for i in f2:
        f1 = pd.merge(f1, i, on='instance_id', how='left')
    return f1


if __name__ == '__main__':
    train_data, test_data = load_train_test_data()
    score = off_test_split(train_data)
    add_feature = [i[0] for i in score[-cross_feature_num:]]
    base_add = pd.concat([base, cross[add_feature]], axis=1)
    LGB_predict(base_add, 'bryan_submit')
