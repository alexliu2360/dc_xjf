from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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


if __name__ == '__main__':
    train_data, test_data = load_train_test_data()
    # 引入train test数据
    train_data, test_data = load_train_test_data()
    train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    test_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    train_data.fillna(-1, inplace=True)
    test_data.fillna(-1, inplace=True)

    label = train_data['Tag']
    train = train_data.drop(['UID', 'Tag'], axis=1)
    test_id = test_data['UID']
    test = test_data.drop(['UID', 'Tag'], axis=1)

    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=0.0, reg_lambda=1, max_depth=-1,
                                   n_estimators=5000, objective='binary', subsample=0.7, colsample_bytree=0.77,
                                   subsample_freq=1, learning_rate=0.01,
                                   random_state=2018, n_jobs=50, min_child_weight=4, min_child_samples=5,
                                   min_split_gain=0)
    skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
    best_score = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test_id.shape[0])

    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                      eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                                (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=50)
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print(best_score)
        oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index],
                                                        num_iteration=lgb_model.best_iteration_)[:, 1]

        test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
        sub_preds += test_pred / 5

    m = tpr_weight_funtion(y_predict=oof_preds, y_true=label)
    print(m[1])
    submit = pd.read_csv('../data/submit_sample.csv')
    submit['Tag'] = sub_preds
    submit.to_csv('../data/sub/baseline_%s.csv' % str(m), index=False)
