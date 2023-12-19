import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("HousePrices1.csv")

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

X = df_scaled[['House Area']]
y = df_scaled['price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_b = np.c_[np.ones((len(X_train), 1)), X_train.values]
X_eval_b = np.c_[np.ones((len(X_eval), 1)), X_eval.values]


theta = np.random.randn(2, 1)

learning_rate = 0.0001

for iteration in range(1000):
    gradients = 2/len(X_train) * X_train_b.T.dot(X_train_b.dot(theta) - y_train.values.reshape(-1, 1))
    theta = theta - (learning_rate * gradients)

X_eval_b = np.c_[np.ones((len(X_eval), 1)), X_eval.values]
predictions = X_eval_b.dot(theta)

rmse = np.sqrt(mean_squared_error(y_eval, predictions))
print("RMSE for Evaluation Set:", rmse)

X_test_b = np.c_[np.ones((len(X_test), 1)), X_test.values]
predictions_test = X_test_b.dot(theta)

rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
print("RMSE for Test Set:", rmse_test)
