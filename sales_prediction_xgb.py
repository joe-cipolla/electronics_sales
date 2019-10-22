import numpy as np
import pandas as pd
import seaborn as sns
from GoogleFreeTrans import Translator
import matplotlib.pyplot as plt
import itertools
from itertools import product
import gc
import random as rd # generating random numbers
import datetime # manipulating date formats
from pandas import Series as Series
import time

from fbprophet import Prophet

# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from pyramid.arima import auto_arima

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

import graphviz

import warnings
warnings.filterwarnings("ignore")


sns.set()

translator = Translator.translator(src='ru', dest='en')

print('')
print('loading original data...')
item_cat_df = pd.read_csv('item_categories.csv', sep=',', index_col=0)
items_df = pd.read_csv('items.csv', sep=',', index_col=0)
sales_train_df = pd.read_csv('sales_train.csv', sep=',', index_col=0)
sample_submission_df = pd.read_csv('sample_submission.csv', sep=',', index_col=0)
shops_df = pd.read_csv('shops.csv', sep=',', index_col=0)
test_df = pd.read_csv('test.csv', sep=',', index_col=0)

eng_item_cat_df = pd.read_csv('eng_item_categories.csv', sep=',', index_col=0)
eng_item_cat_df.columns = ['item_category_id']
eng_shops_df = pd.read_csv('eng_shops.csv', sep=',', index_col=0)


dates = pd.to_datetime(sales_train_df.index, format='%d.%m.%Y')
# pd.DataFrame(dates).to_pickle('dates.pkl')


date_block_num = sales_train_df.iloc[:, 0]
shop_id = sales_train_df.iloc[:, 1]
item_id = sales_train_df.iloc[:, 2]
item_price = sales_train_df.iloc[:, 3]
item_cnt_day = sales_train_df.iloc[:, 4]

sales_train_df2 = pd.DataFrame({'date_block_num': date_block_num, 'shop_id': shop_id,
                                'item_id': item_id, 'item_price': item_price, 'item_cnt_day': item_cnt_day})
sales_train_df2.index = dates
del date_block_num, shop_id, item_id, item_price, item_cnt_day
del sales_train_df


# item_price_98th = sales_train_df2['item_price'].quantile(.98)
# item_cnt_99th = sales_train_df2['item_cnt_day'].quantile(.99)
# trimmed_sales_train_df = sales_train_df2
# trimmed_sales_train_df['item_price'] = np.clip(trimmed_sales_train_df['item_price'], 0, item_price_98th)
# trimmed_sales_train_df['item_cnt_day'] = np.clip(trimmed_sales_train_df['item_cnt_day'], 0, item_cnt_99th)



### aggregate data

# sales = trimmed_sales_train_df
# test = test_df
#
# del sales_train_df2, test_df, trimmed_sales_train_df

print('')
print('aggregating data into months...')
sales = sales_train_df2
test = test_df

del sales_train_df2

# Create grid with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num'] == block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num'] == block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))

# Add the shop_id from the test data and create date_block_num 34
block_num = 34
cur_shops = test['shop_id'].unique()
cur_items = test['item_id'].unique()
grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))

# Turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

# The size of the grid should be the same as the sum of the product of unique `shop_counts` & `item_counts` for
# both the sales & test data
shop_counts = sales.groupby('date_block_num')['shop_id'].nunique()
item_counts = sales.groupby('date_block_num')['item_id'].nunique()
test_shops = test['shop_id'].nunique()
print(shop_counts.dot(item_counts) + test_shops)

# Get aggregated values for (shop_id, item_id, month, price)
gb = sales.groupby(index_cols, as_index=False)['item_cnt_day'].agg('sum')
# Rename column
gb = gb.rename(columns={'item_cnt_day': 'target'})
# Join aggregated data to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

all_data['target'] = np.clip(all_data['target'], 0, 20)  # trim extremes

# add predictor variables
all_data = all_data.merge(items_df, on='item_id', how='left')
all_data = all_data[['date_block_num', 'shop_id', 'item_id', 'item_category_id', 'target']]
all_data = all_data.sort_values(by=['shop_id', 'item_id', 'date_block_num'])

print('')
print('adding Shop lag features...')
## create monthly summed sale count by SHOP
columns = ['month', 'shop', 'shop_and_month_sum']
shop_ids = all_data['shop_id'].unique()
index = np.arange(34*len(shop_ids))
shop_month_sum = pd.DataFrame(index=index, columns=columns)
shop_month_sum = shop_month_sum.fillna(0)
i = 0
for s in shop_ids:
    for m in range(0, 34):
        if i in [1, 100, 500, 1000, 1500, 2000]:
            print(str(i) + ' of ' + str(34*len(shop_ids)))
        sum_target = all_data[all_data['shop_id'] == s][all_data['date_block_num'] == m]['target'].sum()
        shop_month_sum.iloc[i, 0] = m + 1
        shop_month_sum.iloc[i, 1] = s
        shop_month_sum.iloc[i, 2] = sum_target
        i += 1

# create lags
shop_month_sum['sams_T1'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T2'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T3'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T4'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T5'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T6'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T7'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T8'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T9'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T10'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T11'] = np.zeros(len(shop_month_sum))
shop_month_sum['sams_T12'] = np.zeros(len(shop_month_sum))
for s in shop_ids:
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T1'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(1)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T1'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(2)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T1'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(3)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T4'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(4)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T5'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(5)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T6'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(6)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T7'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(7)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T8'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(8)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T9'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(9)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T10'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(10)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T11'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(11)
    shop_month_sum.loc[shop_month_sum.shop == s, 'sams_T12'] = \
        shop_month_sum[shop_month_sum['shop'] == s]['shop_and_month_sum'].shift(12)
shop_month_sum = shop_month_sum.fillna(0)

print('')
print('adding Category lag features...')
## create monthly summed sale count by CATEGORY
columns = ['month', 'category', 'cat_and_month_sum']
cat_ids = all_data.item_category_id.unique()
index = np.arange(34*len(cat_ids))
cat_month_sum = pd.DataFrame(index=index, columns=columns)
cat_month_sum = cat_month_sum.fillna(0)
i = 0
for c in cat_ids:
    for m in range(0, 34):
        if i in [1, 100, 500, 1000, 1500, 2000, 2500]:
            print(str(i) + ' of ' + str(34*len(cat_ids)))
        sum_target = all_data[all_data['item_category_id'] == c][all_data['date_block_num'] == m]['target'].sum()
        cat_month_sum.iloc[i, 0] = m + 1
        cat_month_sum.iloc[i, 1] = c
        cat_month_sum.iloc[i, 2] = sum_target
        i += 1

# create lags
cat_month_sum['cams_T1'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T2'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T3'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T4'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T5'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T6'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T7'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T8'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T9'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T10'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T11'] = np.zeros(len(cat_month_sum))
cat_month_sum['cams_T12'] = np.zeros(len(cat_month_sum))
for c in cat_ids:
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T1'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(1)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T1'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(2)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T1'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(3)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T4'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(4)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T5'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(5)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T6'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(6)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T7'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(7)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T8'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(8)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T9'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(9)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T10'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(10)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T11'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(11)
    cat_month_sum.loc[cat_month_sum.category == c, 'cams_T12'] = \
        cat_month_sum[cat_month_sum['category'] == c]['cat_and_month_sum'].shift(12)
cat_month_sum = cat_month_sum.fillna(0)

print('')
print('Merging lag features into monthly aggregated data...')
# merge new features in with orginal all_data df
shop_month_sum = shop_month_sum.rename(columns={'month': 'date_block_num', 'shop': 'shop_id'})
cat_month_sum = cat_month_sum.rename(columns={'month': 'date_block_num', 'category': 'item_category_id'})
lag_df = cat_month_sum.merge(shop_month_sum)
all_data_w_lags = all_data.merge(lag_df, 'left')
del all_data
gc.collect()
all_data_w_lags = all_data_w_lags.fillna(0)
columns = ['date_block_num', 'shop_id', 'item_id', 'item_category_id', 'shop_and_month_sum', 'sams_T1', 'sams_T2',
           'sams_T3', 'sams_T4', 'sams_T5', 'sams_T6', 'sams_T7', 'sams_T8', 'sams_T9', 'sams_T10', 'sams_T11',
           'sams_T12',
           'cat_and_month_sum', 'cams_T1', 'cams_T2', 'cams_T3', 'cams_T4', 'cams_T5', 'cams_T6', 'cams_T7',
           'cams_T8', 'cams_T9', 'cams_T10', 'cams_T11', 'cams_T12', 'target']
all_data_w_lags = all_data_w_lags[columns]


### if RMSE isn't good enough... create total sales per month by city variable






### XGBoost Classifier

# split up training data into train/test for cv
orig_train = all_data_w_lags[all_data_w_lags['date_block_num'] < 34]
train = orig_train[orig_train['date_block_num'] < 29]
test = orig_train[orig_train['date_block_num'] == 29]
submit_test = all_data_w_lags[all_data_w_lags['date_block_num'] == 34]

del all_data_w_lags
gc.collect()

X_train = train.iloc[:, 0:30]
y_train = train.iloc[:, 30]
X_test = test.iloc[:, 0:30]
y_test = test.iloc[:, 30]

dtrain = xgb.DMatrix(X_train, y_train) # Create our DMatrix to make XGBoost more efficient
dtest = xgb.DMatrix(X_test, y_test)

# cv_params = {'max_depth': [1, 2, 3, 5],
#              'min_child_weight': [1, 3, 5],
#              'learning_rate': [0.1, 0.2, 0.5, 0.8],
#              'n_estimators': [10, 100, 500, 1000, 2000],
#              'seed': [11],
#              'subsample': [0.1, 0.3, 0.5, 0.8, 0.9],
#              'colsample_bytree': [0.2, 0.5, 0.8, 0.9],
#              'gamma': [0, 0.2, 0.5, 0.8, 1],
#              'reg_alpha': [0, 0.2, 0.5, 0.8, 1],
#              'reg_lambda': [0, 0.2, 0.5, 0.8, 1],
#              }

# cv_params = {'max_depth': [1],
#              'min_child_weight': [1],
#              'learning_rate': [0.8],
#              'n_estimators': [10],
#              'seed': [11],
#              'subsample': [0.9],
#              'colsample_bytree': [0.9],
#              'gamma': [0],
#              'reg_alpha': [0],
#              'reg_lambda': [0, 0.5],
#              }
#
# ind_params = {'objective': 'reg:tweedie'}  #{'objective': 'binary:logistic'}  “reg:logistic”, “reg:linear”, “survival:cox”, “multi:softmax”, “multi:softprob”, “reg:tweedie”, 'count:poisson'
# optimized_GBM = RandomizedSearchCV(xgb.XGBClassifier(**ind_params), cv_params,
#                              scoring='accuracy', cv=5, n_jobs=1)
# optimized_GBM.fit(dtrain)


#
# cv_xgb = xgb.cv(params=optimized_GBM.best_params_, dtrain=dtrain, num_boost_round=300, nfold=5,
#                 metrics=['error'], # Make sure you enter metrics inside a list or you may encounter issues!
#                 early_stopping_rounds=100) # Look for early stopping that minimizes error
# cv_xgb.tail(5)
# our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
#               'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}

# final_gb = xgb.train(optimized_GBM.best_params_, dtrain, num_boost_round=500)



param = {'max_depth': 1000, 'silent': 1, 'objective': 'reg:tweedie'}
param['nthread'] = 4
param['eval_metric'] = 'rmse'
param['tree_method'] = 'exact'
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 6
plst = param.items()
bst = xgb.train(plst, dtrain, num_round, evallist)
ypred = bst.predict(dtest)
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
bst.save_model('regtweedie_3.model')
# dump model
bst.dump_model('regtweedie_3.dump.raw.txt')


# load saved model
# bst = xgb.Booster({'nthread': 4})  # init model
# bst.load_model('model.bin')  # load data

# cross validated
# cv_xgb = xgb.cv({'objective':'count:poisson', 'nthread':4, 'eval_metric':'rmse'},
#                 dtrain, num_boost_round=5, nfold=5)



# submission predictions
X_submit_test = submit_test.iloc[:, 0:30]
y_submit_test = submit_test.iloc[:, 30]
d_submit_test = xgb.DMatrix(X_submit_test, y_submit_test)
ypred = bst.predict(d_submit_test, ntree_limit=bst.best_ntree_limit)

y_pred_df = pd.DataFrame(data=ypred,  index=y_submit_test.index)
y_pred_df.columns = ['ypred']
y_pred_df = y_pred_df.join(X_submit_test)
y_pred_df = y_pred_df[['shop_id', 'item_id', 'ypred']]
test_df['ID'] = test_df.index
y_pred_df = y_pred_df.merge(test_df, how='left', on=['shop_id', 'item_id'])
y_pred_df = pd.DataFrame({'ID': y_pred_df['ID'], 'item_cnt_month': y_pred_df['ypred']})
y_pred_df = y_pred_df.sort_values('ID')
y_pred_df.to_csv("regtweedie_3_submission.csv", index=False)

xgb.plot_importance(bst)
plt.savefig('regtweedie_3_importances.pdf')

# xgb.plot_tree(bst)
# plt.savefig('poisson_tree2.pdf')
