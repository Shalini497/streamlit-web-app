import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader as pdr
from keras.models import load_model

# App Title

st.title("Stock Trade Prediction App")

st.sidebar.subheader("Query Parameters")
start = st.sidebar.date_input("Start date", dt.datetime(2010, 1, 1))
end = st.sidebar.date_input("End date", dt.datetime.today())
# end = st.sidebar.date_input("End date", dt.datetime.today())

user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = pdr.DataReader(user_input,'yahoo', start, end)

# Describing Data
st.write(df.describe())

# Visualization
st.header("Closing Price VS Time Chart")
fig = plt.figure(figsize=(12,5))
plt.plot(df.Close)
st.pyplot(fig)

st.header("Closing Price VS Time Chart With 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,5))
plt.plot(df.Close)
plt.plot(ma100,'r')
st.pyplot(fig)



st.header("Closing Price VS Time Chart With 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,5))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)


# Splitting the data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# Load the Model

model = load_model("keras_model")

# testing part
past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)



# Splitting data into test

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test , y_test = np.array(x_test) , np.array(y_test)

# Predict the model
y_predict = model.predict(x_test)

scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_predict = y_predict*scaler_factor
y_test = y_test*scaler_factor


# Visualization
st.subheader("Prediction Vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='original price')
plt.plot(y_predict, 'g', label='predict price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
