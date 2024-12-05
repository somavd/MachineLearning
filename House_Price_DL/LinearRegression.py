from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR 

data = pd.read_csv('train.csv')
data = pd.get_dummies(data)
data = data[data['SalePrice'].notna()]

imputer = KNNImputer(n_neighbors=5)
i_data = imputer.fit_transform(data)
i_data = pd.DataFrame(i_data, columns=data.columns)

Y = i_data['SalePrice']
X = i_data.drop('SalePrice', axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

model = LinearRegression()
svr = SVR(kernel='linear')
model.fit(x_train, y_train)
svr.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_svr = svr.predict(x_test)
y_test = y_test.values

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error     :", mae)
print("Mean Squared Error      :", mse)
print("Root Mean Squred Error  :", np.sqrt(mse))
print("r2 score                :", r2)

mae = mean_absolute_error(y_test, y_pred_svr)
mse = mean_squared_error(y_test, y_pred_svr)
r2 = r2_score(y_test, y_pred_svr)
print("Mean Absolute Error     :", mae)
print("Mean Squared Error      :", mse)
print("Root Mean Squred Error  :", np.sqrt(mse))
print("r2 score                :", r2)