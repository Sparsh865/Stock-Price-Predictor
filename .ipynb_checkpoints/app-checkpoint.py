import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf 
from yahoofinancials import YahooFinancials
from keras.models import load_model 
import streamlit as st

print("Imports successful.")

start = '2010-01-01'
end = '2019-12-31'

st.title("Stock Trend Prediction")
print("Streamlit title set.")

user_input = st.text_input("Enter Stock Ticker" , 'AAPL')
print("User input received:", user_input)

df = yf.download(user_input , start = '2019-01-01' , end = '2023-06-12' , progress = False)

df.to_excel("Dow 30.xlsx")

print("Data downloaded successfully:")
print(df)

# CLeaning the Data 
print("Cleaning the Data")
df = df.fillna(0)
duplicate_columns = df.columns[df.columns.duplicated()]

# Iterate over duplicate columns
for dup_col in duplicate_columns:
    # Get indices of the duplicate columns
    dup_col_indices = df.columns.get_loc(dup_col)
    prev_col_indices = df.columns.get_loc(dup_col) - 1
    
    # Count number of zeros in each duplicate pair
    zeros_count_dup_col = df.iloc[:, dup_col_indices].eq(0).sum()
    zeros_count_prev_col = df.iloc[:, prev_col_indices].eq(0).sum()
    
    # Drop the column with more zeros
    if zeros_count_dup_col > zeros_count_prev_col:
        df.drop(dup_col, axis=1, inplace=True)
    else:
        df.drop(df.columns[prev_col_indices], axis=1, inplace=True)


print(df.shape)

df.to_excel("Dow 30 processed.xlsx")

# Describing Data
st.subheader('Data from 2010 - 2019')
print("Displaying data description.")
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
print("Plotting Closing Price vs Time Chart.")
fig = plt.figure(figsize = (12 , 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
print("Plotting Closing Price vs Time chart with 100MA.")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12 , 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
print("Plotting Closing Price vs Time chart with 200MA.")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12 , 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

print("All visualizations plotted successfully.")


############ SPLITTING DATA FOR TRAINING AND TESTING ############# 
print("Splitting data into training and testing")

data_training = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70) : int(len(df))])

print("Data split into training and testing sets.")

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0 , 1))

data_training_array = scaler.fit_transform(data_training)

print("Data scaled using MinMaxScaler.")

model = load_model("LSTM_model.h5")

print("Loaded the model")

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
final_df = final_df.iloc[:, -1].to_frame()

print("Created final dataframe for testing.")
print(final_df.head())


input_data = scaler.fit_transform(final_df)

scaled_data = scaler.fit_transform(final_df)

# Save the scaled data to a text file
np.savetxt('scaled_data.txt', scaled_data, delimiter='\t')

print("Scaled data saved to scaled_data.txt")

print(input_data.shape)



print("Scaled input data using MinMaxScaler.")

x_test = []
y_test = []

for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i , 0])

x_test , y_test = np.array(x_test)  , np.array(y_test)

print("Testing data prepared successfully.")

print(x_test.shape)


y_predicted = model.predict(x_test)
print(x_test)

print("Predictions made.")

scaler = scaler.scale_ 

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

print("Predictions scaled back to original values.")

st.subheader("Predictions vs Original")
print("Plotting Predictions vs Original graph.")
fig2 = plt.figure(figsize = (12 , 6))
plt.plot(y_test , 'b', label = 'Original Price')
plt.plot(y_predicted , 'r' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

print("Graph plotted successfully.")








