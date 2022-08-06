import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('car_details_f.csv').dropna(axis=0) # from:https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv
num = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 0}
data.owner = [num[item] for item in data.owner]

x = data[['year', 'km_driven', 'owner', 'seats']]
y = data['selling_price']
# train_test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=0)

# regression model
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test) # prediction

pickle.dump(model, open('model.pkl','wb'))

