# Facebook Stock Regression

This simple project works with publicly available data of Facebook Stock Data. It uses scikit-learn along with general Python libraries(pandas,numpy,pickle,matplotlib, etc.) to provide  basic linear regression forecasting  a few days into the future.

The script segments the 66 data points into training and testing data, predicting around 3-4 days ahead per data point. The code should work with any data set as long as necessary adjustments are made to the data segmentationa and the batch sizes. 

The features selected for training were the adjusted close ,the volatility, and the high. The label were obtained through shiting the adjusted close price up for forecasting.

Credits:sentdex(https://www.youtube.com/user/sentdex)
