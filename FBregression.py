#Neel Kattumadam
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing,cross_validation, svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

#import the CSV and drop dates
df=pd.read_csv('HistoricalQuotes.csv')
df=df.drop(['date'],1)

#Create volatility and data prediction subset
df['volatility']=(df['high']-df['low'])/(df['high'])
length=int(math.ceil(0.08*len(df)));
print(length)

#Fill extra columns
df.fillna(-9999,inplace=True)
df['label']=df['close'].shift(-length)
df.dropna(how='any', inplace=True)


#Add data to an array and separate into features and labels
X=np.array(df.drop(['label','volume'],1))
y=np.array(df['label'])
X=preprocessing.scale(X)

#Create a prediction/forecast data set
X_lately=X[-length:]
print((X_lately))
X=X[:-length]
print(X)
y=y[:-length]

#Segment data into training and testing
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)

#Use pickle to save and load the model
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)
pickle_in=open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)

#Generate an accuracy and forecasts
accuracy=clf.score(X_test,y_test)
forecast=clf.predict(X_lately)
print(forecast,accuracy)

#Create columns to store data
counter=-length
df["Forecast"]=np.nan
df["Date"]=np.nan

#Set data in dataframe
count=len(df)
for i in forecast:
    df.set_value(count,"Forecast",i)
    count=count+1
print(df)


#Plot data
df["close"].plot()
df["Forecast"].plot()
plt.xlabel('Days')
plt.legend(loc=3)
plt.ylabel('Price')
plt.title('Facebook Stock Prices Over Time')
plt.show()
