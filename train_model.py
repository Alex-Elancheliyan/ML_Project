
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

#USING PANDAS LOAD DATASET
df = pd.read_csv('UsedCarPrice.csv')


x = df.drop(['Car_Name','Selling_Price'], axis=1 )
y = df['Selling_Price']


x = pd.get_dummies(x, columns=['Fuel_Type','Seller_Type','Transmission'], drop_first=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


my_lr_model = LinearRegression()


my_lr_model.fit(x_train, y_train)


with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(my_lr_model, file)

with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

y_pred = loaded_model.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
score = r2_score(y_test, y_pred)
print('R-Squared Value is:',score)
