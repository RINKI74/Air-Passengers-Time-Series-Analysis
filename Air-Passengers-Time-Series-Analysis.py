#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# load the dataset

data=pd.read_csv('AirPassengers.csv')


# In[3]:


# Data Analysis and cleaning 

data.shape


# In[4]:


data.describe()


# In[5]:


data.head()


# In[6]:


# Create the 'Date' as Index for data and viewing the dataset

data['Month']=pd.to_datetime(data['Month'], infer_datetime_format=True)
data=data.set_index(['Month'])
print(data.head())


# In[7]:


#checking and analysing the data

plt.figure(figsize=(20,10))
plt.xlabel("Month")
plt.ylabel("Number of Air Passengers")
plt.plot(data)


# From the above below, we can see that there is a Trend compoenent in the series. Hence, we now check for stationarity of the data.
# 
# Let's make one function consisting of stationary data checking and ADCF test. 

# In[ ]:





# In[8]:


#Stationarity check

rolmean=data.rolling(window=12).mean()
rolstd=data.rolling(window=12).std()
print(rolmean.head(15))
print(rolstd.head(15))


# mean and variance is not constant

# In[9]:


plt.figure(figsize=(20,10))
actual=plt.plot(data, color='red', label='Actual')
mean_6=plt.plot(rolmean, color='green', label='Rolling Mean') 
std_6=plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# #ADF (Augmented Dickey-Fuller Test) to check stationarity,
# #Null hypothesis - Time Series is non-stationary
# 

# In[10]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA


# In[13]:


data.rename(columns={'#Passengers':'Passengers'},inplace=True)
data.head()


# In[14]:


from statsmodels.tsa.stattools import adfuller
print('Dickey-Fuller Test: ')
dftest=adfuller(data['Passengers'], autolag='AIC')
dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# Let's rename "#Passengers", seems really annoying the column name.

# In[15]:


data.rename(columns={'#Passengers':'Passengers'},inplace=True)


# In[16]:


data.head()


# Now perform ADF Test to check stationarity

# In[17]:


from statsmodels.tsa.stattools import adfuller
print('Dickey-Fuller Test: ')
dftest=adfuller(data['Passengers'], autolag='AIC')
dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# We fail to reject the null hypothesis, since p-value is greater than 0.05
# Now we take log transformation to make our Time series stationary and plotted visual for it
# We found graph upward trending over time with seasonality

# In[18]:


plt.figure(figsize=(20,10))
data_log=np.log(data)
plt.plot(data_log)


# In[19]:


plt.figure(figsize=(20,10))
MAvg=data_log.rolling(window=12).mean()
MStd=data_log.rolling(window=12).std()
plt.plot(data_log)
plt.plot(MAvg, color='blue')


# From above plot, we can see that Rolling Mean itself has a trend component even though Rolling Standard Deviation is fairly constant with time.
# 
# For time series to be stationary, we need to ensure that both Rolling Mean and Rolling Standard Deviation remain fairly constant WRT time.
# 
# Both the curves needs to be parallel to X-Axis, in our case it is not so.
# 
# We've also conducted the ADCF ie Augmented Dickey Fuller Test. Having the Null Hypothesis to be Time Series is Non Stationary.

# In[20]:


# Differencing

data_log_diff=data_log-MAvg
data_log_diff.head(12)


# In[21]:


# Drop null values

data_log_diff=data_log_diff.dropna()
data_log_diff.head()


# We can apply some sort of transformation to make the time-series stationary. 
# These transformation may include:
# 
# Differencing the Series (once or more)
# Take the log of the series
# Take the nth root of the series
# Combination of the above
# 
# The most commonly used and convenient method to stationarize the series is by differencing the series at least once until it becomes approximately stationary.
# 
#  How to do differencing :
# 
# If Y_t is the value at time t, then the first difference of Y = Yt – Yt-1. 
# In simpler terms, differencing the series is nothing but subtracting the next value by the current value.
# 
# If the first difference doesn’t make a series stationary, we can go for the second differencing and so on.
# 
# For example, consider the following series: [1, 5, 2, 12, 20]
# First differencing gives: [5-1, 2-5, 12-2, 20-12] = [4, -3, 10, 8]
# Second differencing gives: [-3-4, -10-3, 8-10] = [-7, -13, -2]

# In[22]:


def stationarity(timeseries):
    
    rolmean=timeseries.rolling(window=12).mean()
    rolstd=timeseries.rolling(window=12).std()
    
    plt.figure(figsize=(20,10))
    actual=plt.plot(timeseries, color='red', label='Actual')
    mean_6=plt.plot(rolmean, color='green', label='Rolling Mean') 
    std_6=plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Dickey-Fuller Test: ')
    dftest=adfuller(timeseries['Passengers'], autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[23]:


stationarity(data_log_diff)


# 
# From Rolling method, we see that Mean and standard deviation is not varying.
# 
# From ADF, we reject the null hypothesis bcoz p-value is less than 0.05 (significance level)
# 
# Applying all the transformation and methods, our differenced data is now stationary

# In[24]:


plt.figure(figsize=(20,10))
exp_data=data_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(data_log)
plt.plot(exp_data, color='black')


# In[25]:


# Applying differencing on log data as it is not stationary

exp_data_diff=data_log-exp_data
stationarity(exp_data_diff)


# In[26]:


plt.figure(figsize=(20,10))
data_shift=data_log-data_log.shift()
plt.plot(data_shift)


# In[27]:


data_shift=data_shift.dropna()
stationarity(data_shift)


# Trend is stationary.
# 
# Now Decompose the data
# 
# Once, we separate our the components, we can simply ignore trend & seasonality and check on the nature of the residual part.

# In[ ]:





# In[28]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomp=seasonal_decompose(data_log)

trend=decomp.trend
seasonal=decomp.seasonal
residual=decomp.resid

plt.subplot(411)
plt.plot(data_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# There can be cases where an observation simply consist of trend & seasonality. In that case, there won't be any residual component & that would be a null or NaN. Hence, we also remove such cases.

# In[29]:


from statsmodels.tsa.stattools import acf, pacf

lag_acf=acf(data_shift, nlags=20)
lag_pacf=pacf(data_shift, nlags=20, method='ols')

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.title('Partial Autocorrelation Function')


# From the ACF graph, we can see that curve touches y=0.0 line at x=2. Thus, from theory, Q = 2 From the PACF graph, we see that curve touches y=0.0 line at x=2. Thus, from theory, P = 2
# 
# ARIMA is AR + I + MA. Before, we see an ARIMA model, let us check the results of the individual AR & MA model. Note that, these models will give a value of RSS. Lower the RSS values indicates a better model.

# In[49]:


import warnings

warnings.filterwarnings("ignore")
import seaborn as sns
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# In[51]:


#Modeling
# Build Model
model = ARIMA(data_log, order=(1,0,1), seasonal_order=(1,0,1,12))  
fitted = model.fit()  
print(fitted.summary())


# In[59]:


# Forecast
fc = fitted.forecast(len((data_log)))


# In[61]:


fc_series = pd.Series(fc, index=data_log.index)

plt.figure(figsize=(15,10))

plt.plot(data_log, color = 'blue', label='Actual Test data')

plt.plot(fc_series, color = 'orange',label='Predicted test data')

plt.title('# of Passenger prediction')
plt.xlabel('Time')
plt.ylabel('# of Passengers')
plt.legend(loc='upper left', fontsize=8)

