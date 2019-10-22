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

import gc

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
transactions.plot.line(title='Item Sales over time')
plt.ylabel('Number of Transactions')
plt.xlabel('Month Number')
plt.savefig('Total_Transactions_by_Month.pdf')
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
plt.axes().yaxis.set_major_locator(plt.NullLocator())
plt.xlabel('Number of Monthly Item Sales')
plt.title('Less Than 10 Sales - Density Plot')
plt.savefig('Monthly_Item_Sales_Density_Plot.pdf')
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



# pick best order by aic
# smallest aic value wins
best_aic = np.inf
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(arma22, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))


#
# pick best order by aic
# smallest aic value wins
best_aic = np.inf
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(new_ts.values, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))


# adding the dates to the Time-series as index
ts = sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index = pd.date_range(start = '2013-01-01', end='2015-10-01', freq = 'MS')
ts = ts.reset_index()
ts.head()



# Prophet GAM -- prophet reqiures a pandas df at the below config
# ( date column named as DS and the value column as Y)
ts.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly
model.fit(ts) #fit the model with your dataframe

# predict for five months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 5, freq = 'MS')
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast)
plt.title('Prophet GAM Forecast')
plt.show()

model.plot_components(forecast)
plt.show()



# UCM - Unobserved Components Model
total_sales=sales.groupby(['date_block_num'])["item_cnt_day"].sum()
dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

total_sales.index=dates
total_sales.head()

# get the unique combinations of item-store from the sales data at monthly level
monthly_sales=sales.groupby(["shop_id","item_id","date_block_num"])["item_cnt_day"].sum()
# arrange it conviniently to perform the hts
monthly_sales=monthly_sales.unstack(level=-1).fillna(0)
monthly_sales=monthly_sales.T
dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
monthly_sales.index=dates
monthly_sales=monthly_sales.reset_index()
monthly_sales.head()


# Bottoms up
# Calculating the base forecasts using prophet
# From HTSprophet pachage -- https://github.com/CollinRooney12/htsprophet/blob/master/htsprophet/hts.py
start_time = time.time()
forecastsDict = {}
for node in range(len(monthly_sales)):
    # take the date-column and the col to be forecasted
    nodeToForecast = pd.concat([monthly_sales.iloc[:,0], monthly_sales.iloc[:, node+1]], axis = 1)
#     print(nodeToForecast.head())  # just to check
# rename for prophet compatability
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)
    if (node== 10):
        end_time=time.time()
        print("forecasting for ", node, "th node and took", end_time-start_time, "s")
        break


# Middle Out - predict store level
monthly_shop_sales = sales.groupby(["date_block_num", "shop_id"])["item_cnt_day"].sum()
# get the shops to the columns
monthly_shop_sales = monthly_shop_sales.unstack(level=1)
monthly_shop_sales = monthly_shop_sales.fillna(0)
monthly_shop_sales.index = dates
monthly_shop_sales = monthly_shop_sales.reset_index()
monthly_shop_sales.head()

start_time = time.time()
# Calculating the base forecasts using prophet
# From HTSprophet pachage -- https://github.com/CollinRooney12/htsprophet/blob/master/htsprophet/hts.py
forecastsDict = {}
for node in range(len(monthly_shop_sales)):
    # take the date-column and the col to be forecasted
    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:, 0], monthly_shop_sales.iloc[:, node + 1]], axis=1)
    #     print(nodeToForecast.head())  # just to check
    # rename for prophet compatability
    nodeToForecast = nodeToForecast.rename(columns={nodeToForecast.columns[0]: 'ds'})
    nodeToForecast = nodeToForecast.rename(columns={nodeToForecast.columns[1]: 'y'})
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods=1, freq='MS')
    forecastsDict[node] = m.predict(future)

# predictions = np.zeros([len(forecastsDict[0].yhat),1])
nCols = len(list(forecastsDict.keys()))+1
for key in range(0, nCols-1):
    f1 = np.array(forecastsDict[key].yhat)
    f2 = f1[:, np.newaxis]
    if key==0:
        predictions=f2.copy()
       # print(predictions.shape)
    else:
       predictions = np.concatenate((predictions, f2), axis = 1)

predictions_unknown=predictions[-1]
predictions_unknown



# ARIMA
### from Digital Ocean  https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

# ARIMA parameter selection
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(total_sales,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# best result: ARIMA(3, 3, 3)x(2, 0, 0, 12)12 - AIC:106.29837434086629

# fit ARIMA
mod = sm.tsa.statespace.SARIMAX(total_sales,
                                order=(2, 2, 0),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])

# diagnostics
results.plot_diagnostics()
plt.show()

# predict
pred = results.get_prediction(start=pd.to_datetime('2015-11-01'), dynamic=False)
pred_ci = pred.conf_int()




### from statsmodels  ---  http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html
d_total_sales = total_sales.diff()
fig, axes = plt.subplots(1, 2, figsize=(15,4))
# Levels
axes[0].plot(total_sales.index._mpl_repr(), total_sales, '-')
axes[0].set(title='Total Montly Sales')
# Log difference
axes[1].plot(total_sales.index._mpl_repr(), d_total_sales, '-')
axes[1].hlines(0, total_sales.index[0], total_sales.index[-1], 'r')
axes[1].set(title='Total Monthly Sales - differenced');
plt.show()

# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(d_total_sales.ix[1:, ], lags=20, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(d_total_sales.ix[1:, ], lags=20, ax=axes[1])
plt.show()

# Fit the model
mod = sm.tsa.statespace.SARIMAX(d_total_sales, trend='c', order=(1,1,1))
res = mod.fit(disp=False)
print(res.summary())

# predict
# In-sample one-step-ahead predictions
predict = res.get_prediction()
predict_ci = predict.conf_int()



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




#
#
# ###################################################################################
# ### XGBoost Classifier

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
# print('')