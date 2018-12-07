{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import lightgbm as lgb\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "def tpr_weight_funtion(y_true,y_predict):\n",
    "    d = pd.DataFrame()\n",
    "    d['prob'] = list(y_predict)\n",
    "    d['y'] = list(y_true)\n",
    "    d = d.sort_values(['prob'], ascending=[0])\n",
    "    y = d.y\n",
    "    PosAll = pd.Series(y).value_counts()[1]\n",
    "    NegAll = pd.Series(y).value_counts()[0]\n",
    "    pCumsum = d['y'].cumsum()\n",
    "    nCumsum = np.arange(len(y)) - pCumsum + 1\n",
    "    pCumsumPer = pCumsum / PosAll\n",
    "    nCumsumPer = nCumsum / NegAll\n",
    "    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]\n",
    "    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]\n",
    "    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]\n",
    "    return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import os\n",
    "data_path = '../data/'\n",
    "origin_path = data_path + 'origin_data/'\n",
    "\n",
    "op_train_new_fn = 'operation_train_new.csv'\n",
    "tran_train_new_fn = 'transaction_train_new.csv'\n",
    "tag_train_new_fn = 'tag_train_new.csv'\n",
    "op_train_sorted_fn = 'op_train_sorted.csv'\n",
    "tran_train_sorted_fn = 'tran_train_sorted.csv'\n",
    "tag_train_sorted_fn = 'tag_train_sorted.csv'\n",
    "\n",
    "op_origin_fn = 'op_origin.csv'\n",
    "tran_origin_fn = 'tran_origin.csv'\n",
    "\n",
    "\n",
    "op_train_new_file = data_path + op_train_new_fn\n",
    "tran_train_new_file = data_path + tran_train_new_fn\n",
    "tag_train_new_file = data_path + tag_train_new_fn\n",
    "op_train_sorted_file = data_path + op_train_sorted_fn\n",
    "tran_train_sorted_file = data_path + tran_train_sorted_fn\n",
    "tag_train_sorted_file = data_path + tag_train_sorted_fn\n",
    "\n",
    "op_origin_file = origin_path + op_origin_fn\n",
    "tran_origin_file = origin_path + tran_origin_fn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def load_data():\n",
    "    print('[info]:start read from op_train...')\n",
    "    op_train = pd.read_csv(op_origin_file).drop('Unnamed: 0', axis=1)\n",
    "    print('[info]:start read from tran_train...')\n",
    "    tran_train = pd.read_csv(tran_origin_file).drop('Unnamed: 0', axis=1)\n",
    "    print('[info]:start read from tag_train...')\n",
    "    tag_train = pd.read_csv(tag_train_sorted_file).drop('Unnamed: 0', axis=1)\n",
    "    return op_train, tran_train, tag_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_tag(uid):\n",
    "    return tag_train[tag_train['UID'] == uid]\n",
    "\n",
    "def get_op(uid):\n",
    "    return op_train[op_train['UID'] == uid]\n",
    "\n",
    "def get_tran(uid):\n",
    "    return tran_train[tran_train['UID'] == uid]\n",
    "\n",
    "def get_value_counts(uid, train_data):\n",
    "    assert type(train_data) is pd.DataFrame\n",
    "    for c in list(train_data.columns):\n",
    "        print('[%r]'%c)\n",
    "        print(train_data[train_data['UID']==uid][c].value_counts())\n",
    "        print('====')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "op_train, tran_train, tag_train = load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "op_gb = op_train.groupby('UID')\n",
    "drop_fts = ['mode','success','time','device1','device2','device_code1','device_code2','device_code3','mac1','ip1','ip2','ip1_sub','ip2_sub','timestamp']\n",
    "op_train_nf = op_gb.count().drop(drop_fts, axis='columns')\n",
    "op_train_nf.rename(columns={'day':'day_cnts'}, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "op_10001 = get_op(10001)\n",
    "op_10001.T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(op_10001['timestamp'].get_values()[0])\n",
    "print(op_10001['timestamp'].get_values()[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "op_10001_gb = op_10001.groupby('day')\n",
    "op_10001_gb.indices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "op_10001_gb = op_10001.groupby('day')\n",
    "for gb_key in op_10001_gb.indices.keys():\n",
    "    diff_time = []\n",
    "    sgb = op_10001_gb.get_group(gb_key)\n",
    "    timestamp = sgb['timestamp'].get_values()\n",
    "    t_l = list(timestamp)\n",
    "    print(len(t_l))\n",
    "#     diff_time = [ t_l[t_l.index(t)+1] - t ]\n",
    "    for t in t_l:\n",
    "        if t_l.index(t)<=len(t_l):\n",
    "            print(t_l.index(t))\n",
    "        else:\n",
    "            print(t_l.index(t))\n",
    "#     diff_time = [ t_l[t_l.index(t)+1] - t for t in t_l if t_l.index(t)<=len(t_l)]\n",
    "    print(diff_time, '===')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "feature['day_cnts'] = op_10001['day'].count()\n",
    "feature['top_appear_day'] = op_10001['day'].value_counts().sort_values(ascending=False).index[0]\n",
    "feature['top_appear_day_cnt'] = op_10001['day'].value_counts().sort_values(ascending=False).values[0]\n",
    "feature['op_times_per2min'] = op_10001['day'].value_counts().sort_values(ascending=False).values[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "op_train_nf = op_train_nf.join(day_cnt)\n",
    "op_train_nf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_feature(op,trans,label):\n",
    "    for feature in op.columns[2:]:\n",
    "        label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')\n",
    "        label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')\n",
    "    \n",
    "    for feature in trans.columns[2:]:\n",
    "        if trans_train[feature].dtype == 'object':\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')\n",
    "        else:\n",
    "            print(feature)\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')\n",
    "            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')\n",
    "    return label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train = get_feature(op_train,trans_train,y)\n",
    "test = get_feature(op_test,trans_test,sub)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "list(train.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train = train.fillna(-1)\n",
    "test = test.fillna(-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train = train.drop(['UID','Tag'],axis = 1).fillna(-1)\n",
    "label = y['Tag']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "test_id = test['UID']\n",
    "test = test.drop(['UID','Tag'],axis = 1).fillna(-1)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
