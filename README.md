# OPTIVER-REALIZED-VOLATILITY-PREDICTION

This is from Kaggle Competition:  [Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction)

### Problem 
The goal is to predict the realized volatility of the next 10 minutes given the book data and trade data of the first 10 minutes.

### Structure of the project
- We use numpy/pandas/tensorflow when preprocessing these data. See [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/DataProcessing.py). 
- We will add a version of data preprocessing using pyspark.  
- We first test our model using linear regression with cross validation. See [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/BestLinearRegression.py) for code and [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/LinearRegressionResults.csv) for results. 
- The best model we have so far is described in below and can be seen [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/TestGlobalAndLinear.py), also see [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/4_inputs_results.csv) for results.
- We use automated hyperparameter tuning techniques provided by tensorboard.plugins.hparams, see [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/hyperparametersTuning.py).
- We also add some experiment with detecting market abnormality using both linear model and cnn model. See the end of [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/TestGlobalAndLinear.py). The hope is that by adding these abnormality detection we can split our data into 'normal' and 'abnormal', and build a model for each. We only notice slight improvement using this method, but it seems interesting. 

### Data
We are given the book data and trade data of over 100 stocks, a total 3GB size. 
The book data is a dataframe with
- stock_id, time_id
- seconds_in_bucket (range from 0-599)
- bid_price[1/2], ask_price[1/2], bid_size[1/2], ask_size[1/2]

Trade data is another dataframe with 
- stock_id, time_id
- seconds_in_bucket (range from 0-599)
- price, size, order_count 

Target data is in train.csv. It it a dataframe with three columns: 
- stock_id, time_id 
- target

More specifically, each time_id corresponds to a 20 minutes interval, with the first 10 minutes data given. seconds_in_bucket range from 0 to 599 (600 seconds in total). Out job is to use 

### Data Preprocessing
Basically, for each stock_id and time_id, we have some seconds_in_bucket. For seconds_in_bucket we have book_data (bid_price[1/2], ask_price[1/2], bid_size[1/2], ask_size[1/2]) and trade data (price, size, order_count). 

Since a lot of seconds_in_bucket are missing as data remains the same during that second, we need to fill in missing data using fillna('ffill'). 

Then we extract features, these features are
- WAP[1/2], which is the weighted average of bid_price[1/2], ask_price[1/2], bid_size[1/2], ask_size[1/2]
- seconds by seconds log return. 
- for 600 seconds of data, we split into 10*60 seconds, and for each 60 seconds interval we compute it's realized volatility (std of log return)
- weighted realized volatility. We have several version of this, e.g. (60 seconds realized volatility) * (bid/ask size during this 60 seconds). As it turns out, one of the best features in this category is (60 seconds realized volatility) * log(bid/ask size during this 60 seconds)). 
- Now for each time_id, we have 600 seconds* N features data, where N can range from 3-10, depending on how many features we want to include. 

### Model Architecture
We tried various models. The best model architecture based purely on DNN so far is the following:

We build a 4 inputs model, where for each time_id 
- input1 is of size 10 (10 * 60 seconds each interval) * N (features), N in range(3,11). We then pass this input into a convolution network ( about 5-7 layers, with multiple convolution layers + multiple dense layers + regularization layers). The idea is that by reducing the time dimension from 600 seconds to 10 * 60 seconds, convolution layers should be able to catch a lot of information (comparing to recurssion layers). 
- input2 is of size 600 (seconds) * 3 (features: log return of WAP[1/2], log return of (trade) price). We then pass this input into a recurrent network (about 5-7 layers, with multiple RNN/LSTM layers followed by dense layers). The idea is that for a time series with 600 seconds, recurrent network might perform better than others.
- input3 is of size 10 (10 intervals each with 60 seconds) * M features, M in range (3,11). Then we pass this input into a very simple Dense layer. This simple Dense layer alone is a simple linear regression. The idea here is this is the 'backbone' of our model: all the other layers can be thought of as 'refining this simple linear regression'.
- input4 is of size 10 ( 10 intervals each 60 seconds) * 112 (total number of stocks) * K features. We then process this input with a convolution layer similar to before. This include the information of the overall market. In fact adding this input can significantly improve our model performance. 

We then concatenate these four networks together and output a single number (the target we predict). 

### Model Evaluation
The metric function is MSE. A model with only input3 (simple linear regression) can achieve average R2 score of around 0.73(see [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/LinearRegressionResults.csv)). Model with input1, input2 and input3 can achieve R2 score around 0.8, and model with input1-input4 can further improve the result to around 0.85 (see [here](https://github.com/taosongst/OPTIVER-REALIZED-VOLATILITY-PREDICTION/blob/main/4_inputs_results.csv)).

