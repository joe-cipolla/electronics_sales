import numpy as np
import pandas as pd
import seaborn as sns
from GoogleFreeTrans import Translator
import matplotlib.pyplot as plt
import itertools
from itertools import product
import random as rd # generating random numbers
import datetime # manipulating date formats
from pandas import Series as Series
import time

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

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

import warnings
warnings.filterwarnings("ignore")


# ### setup R env in Python
#
# # import R's "base" package
# base = importr('base')
# # import R's utility package
# utils = rpackages.importr('utils')
# # select a mirror for R packages
# utils.chooseCRANmirror(ind=1) # select the first mirror in the list
# # R package names
# packnames = ('fpp', 'forecast')
# # R vector of strings
# from rpy2.robjects.vectors import StrVector
# # Selectively install what needs to be install.
# if len(packnames) > 0:
#     utils.install_packages(StrVector(packnames))



# predict total sales for every product and store for the next month


sns.set()

translator = Translator.translator(src='ru', dest='en')


item_cat_df = pd.read_csv('item_categories.csv', sep=',', index_col=0)
items_df = pd.read_csv('items.csv', sep=',', index_col=0)
sales_train_df = pd.read_csv('sales_train.csv', sep=',', index_col=0)
sample_submission_df = pd.read_csv('sample_submission.csv', sep=',', index_col=0)
shops_df = pd.read_csv('shops.csv', sep=',', index_col=0)
test_df = pd.read_csv('test.csv', sep=',', index_col=0)

eng_item_cat_df = pd.read_csv('eng_item_categories.csv', sep=',', index_col=0)
eng_item_cat_df.columns = ['item_category_id']
eng_shops_df = pd.read_csv('eng_shops.csv', sep=',', index_col=0)

sales_train_df.describe()
sales_train_df.head()
sales_train_df.tail()
sales_train_df.isnull().sum()


###################################################################################
translate = 0
if translate == 1:

    ### Translate to English
    eng_shops = []
    for i in range(len(shops_df)):
        eng_shops.append(translator.translate(shops_df.iloc[i].name))
    eng_shops_df = shops_df['shop_id']
    eng_shops_df.index = eng_shops
    eng_shops_df.to_csv('eng_shops.csv')

    eng_cat = []
    for i in range(len(item_cat_df)):
        eng_cat.append(translator.translate(item_cat_df.iloc[i].name))
    eng_item_cat_df = item_cat_df['item_category_id']
    eng_item_cat_df.index = eng_cat
    eng_item_cat_df.to_csv('eng_item_categories.csv')
###################################################################################



###################################################################################
### data formatting

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



###################################################################################
### EDA

sales_train_df2.isnull().sum()
sales_train_df2.isna().sum()


sales_train_df2['item_id'].plot.hist()
plt.title('Daily Item_ID Histogram')
plt.show()

sns.distplot(sales_train_df2['item_price'])
plt.title('Daily Item Price Density Plot')
plt.show()

sales_train_df2[sales_train_df2['item_price'] > 10000]

sns.distplot(sales_train_df2['item_price'])  # item ID range matches items.csv file
plt.title('Daily Item Price Density Plot')
plt.show()


really_expensive_items = sales_train_df2[sales_train_df2['item_price'] > 10000]['item_id']
really_expensive_items = pd.DataFrame(really_expensive_items).merge(items_df, on='item_id', how="right")
eng_item_cat_df['item_category_name'] = eng_item_cat_df.index
really_expensive_items = really_expensive_items.merge(eng_item_cat_df, on='item_category_id', how="right")
really_expensive_items['item_category_name'].unique()  # these $10k items are in categories like CDs, Books, etc
really_expensive_items.groupby('item_category_name').item_id.nunique()


more_than_200_sold_in_a_day = sales_train_df2[sales_train_df2['item_cnt_day'] > 200]['item_id']
for i in more_than_200_sold_in_a_day:
    print(translator.translate(items_df[items_df['item_id'] == i].index[0]))
# only 6 items sold more than 600 in a single day: corporate t-shirts, grand theft auto, comicon tickets
# it kinda makes sense to eliminate these, the comicon ended in 2015, but since we are predicting by item, this might
# be good info about each product, it is reasonable that they would sell this much in one day
# BUT since we are predicting by product, and most of these high selling video games and event tickets are specific
# to that issue of the videogame/event, they probably won't produce as many sales the next year (comincon 2015 only happens once)


sales_train_df2[sales_train_df2['item_price'] > 10000]['shop_id'].unique() # these outliers aren't all coming from the same shop(s)
# they really seem to be all over the place, should probably just take them out



transactions = sales_train_df2.groupby('date_block_num')['item_cnt_day'].count()
transactions.plot.line(title='Number of transactions by month')
plt.show()


transactions = sales_train_df2[sales_train_df2['date_block_num']==10].groupby('shop_id').count()
transactions.plot.line(title='November Number of transactions by Store')
plt.show()


# do we even need pricing data?, shouldn't be a big deal to clip it

# cap extreme prices
item_price_98th = sales_train_df2['item_price'].quantile(.98)
item_cnt_99th = sales_train_df2['item_cnt_day'].quantile(.99)
trimmed_sales_train_df = sales_train_df2
trimmed_sales_train_df['item_price'] = np.clip(trimmed_sales_train_df['item_price'], 0, item_price_98th)
trimmed_sales_train_df['item_cnt_day'] = np.clip(trimmed_sales_train_df['item_cnt_day'], 0, item_cnt_99th)


# lets try this again
sns.distplot(trimmed_sales_train_df['item_price'])
plt.title('Capped Daily Item Price Density Plot')
plt.show()


sns.distplot(sales_train_df2['item_cnt_day'])
plt.title('Item Daily Count Density Plot')
plt.show()



# log_item_price = np.log(sales_train_df2[sales_train_df2['item_price'] < 8000]['item_price']).fillna(0)
# log_item_price.isna().sum()



###################################################################################
### aggregate data

sales = trimmed_sales_train_df
test = test_df

del sales_train_df2, test_df, trimmed_sales_train_df

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



###################################################################################
# aggregated data EDA

sns.distplot(all_data['target'])
plt.title('Monthly Target Density Plot')
plt.show()

less_than10 = all_data[all_data['target'] < 10]
sns.distplot(less_than10['target'])
plt.title('Less Than 10 Target Density Plot')
plt.show()  # most values are zero

all_data.to_csv('all_data.csv', index=False)
all_data.info()
all_data.head()
all_data.describe()

# number of items per cat
x = items_df.groupby(['item_category_id']).count()
x = x.sort_values(by='item_id', ascending=False)
x = x.iloc[0:30].reset_index()
x
# #plot
plt.figure()
ax = sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category (Top 30 Categories by Count Size)")
plt.ylabel('Number of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()

top30cats = pd.DataFrame(x.iloc[:, 0])
eng_item_cat_df['category'] = eng_item_cat_df.index
top30cats = top30cats.merge(eng_item_cat_df, on="item_category_id")
print(top30cats.sort_values(by="category"))

# are all items in test set also in training set?
train_unique_items = sales['item_id'].unique()
test_unique_items = test['item_id'].unique()
missing_items = 0
for i in test_unique_items:
    if i in train_unique_items:
        pass
    else:
        missing_items += 1
# there are   missing_items = 363


###################################################################################
### time series decomposition

ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of All Stores')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.show()

# multiplicative
res = sm.tsa.seasonal_decompose(ts.values, freq=12, model="multiplicative")
plt.title("Multiplicative Decomposition")
fig = res.plot()
fig.show()

# Additive model
res = sm.tsa.seasonal_decompose(ts.values, freq=12, model="additive")
plt.title("Additive Decomposition")
fig = res.plot()
fig.show()


# Stationarity tests
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(ts)


# to remove trend
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob

ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16, 16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts = difference(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts = difference(ts, 12)       # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()

test_stationarity(new_ts)


###################################################################################
### AR, MA and ARMA models

def tsplot(y, lags=None, figsize=(10, 8), style='bmh', title=''):

    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


# Simulate an AR(1) process with alpha = 0.6
np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]
limit = 12
_ = tsplot(x, lags=limit, title="AR(1)process")
plt.show()


# Simulate an AR(2) process
n = int(1000)
alphas = np.array([.444, .333])
betas = np.array([0.])

# Python requires us to specify the zero-lag value which is 1
# Also note that the alphas for the AR model must be negated
# We also set the betas for the MA equal to 0 for an AR(p) model
# For more information see the examples at statsmodels.org
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ar2, lags=12,title="AR(2) process")
plt.show()


# Simulate an MA(1) process
n = int(1000)
# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.8])
# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
limit = 12
_ = tsplot(ma1, lags=limit,title="MA(1) process")
plt.show()


# Simulate an ARMA(2, 2) model with alphas=[0.5,-0.25] and betas=[0.5,-0.3]
max_lag = 12

n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit

alphas = np.array([0.8, -0.65])
betas = np.array([0.5, -0.7])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
_ = tsplot(arma22, lags=max_lag,title="ARMA(2,2) process")
plt.show()




# UCM - Unobserved Components Model
total_sales=sales.groupby(['date_block_num'])["item_cnt_day"].sum()
dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

total_sales.index=dates
total_sales.head()


### from pyramid   ---  THIS WORKS THE BEST
stepwise_fit = auto_arima(total_sales, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0,
                          seasonal=True, d=1, D=1, trace=True, error_action='ignore',
                          suppress_warnings=True, stepwise=True)

stepwise_fit.summary()
residuals = pd.Series(stepwise_fit.resid())
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
residuals.describe()
stepwise_fit.arima_res_.plot_diagnostics()
plt.show()

stepwise_fit.predict(1)

