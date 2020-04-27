import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def visualize_testing(y_test, y_predictions):
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color='red', label='Real Google Stock Price')
    plt.plot(y_predictions, color='blue', label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction - TESTING')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()


def visualize_new_prediction(y_predictions):
    plt.figure(figsize=(14, 5))
    plt.plot(y_predictions, color='blue', label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction - PREDICTION')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Load the data
    data = pd.read_csv('./../data/GOOG.csv', date_parser=True)

    # Everything before 2020 is training data. Everything in 2020 is test data.
    # Drop date and adj close columns as they don't have useful info.
    training_data = data[data['Date'] < '2019-01-01'].copy().drop(['Date', 'Adj Close'], axis=1)
    test_data = data[data['Date'] >= '2019-01-01'].copy()

    # Data in a max value of 1 and min value of 0.
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    y_train = []

    # Divide the data into 60 day chunks, train and predict the 61th day. And keep doing that for all data.
    for i in range(60, training_data.shape[0]):
        X_train.append(training_data[i - 60: i])
        y_train.append(training_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Build the model.
    # Return sequences returns sequence from first layer to second layer.
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(units=80, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(units=120, activation='relu'),
        tf.keras.layers.Dropout(0),

        tf.keras.layers.Dense(units=1)
    ])

    # model.compile(optimizer='adam', loss='mean_squared_error')
    #
    # # Train the model
    # model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Testing the model
    training_data = data[data['Date'] < '2019-01-01'].copy()
    test_data = data[data['Date'] >= '2019-01-01'].copy()

    past_60_days = training_data.tail(60)
    df = past_60_days.append(test_data, ignore_index=True).drop(['Date', 'Adj Close'], axis=1)
    inputs = scaler.transform(df)

    X_test = []
    y_test = []

    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60: i])
        y_test.append(inputs[i, 0])  # Opening price of the stock

    X_test, y_test = np.array(X_test), np.array(y_test)

    y_predictions = model.predict(X_test)

    # Inverse scaling.
    inv_scale = 1 / 8.18605127e-04
    y_predictions = y_predictions * inv_scale
    y_test = y_test * inv_scale

    visualize_testing(y_test, y_predictions)

    # Predict tomorrow's stock by passing data of last 60 days.




