import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle

# #Index(['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
#        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
#        'parking', 'prefarea', 'furnishingstatus'],
#       dtype='object')


raw_df = pd.read_csv('Housing.csv')

raw_df['guestroom']=raw_df['guestroom'].map({'yes':1,'no':0})
raw_df['mainroad']=raw_df['mainroad'].map({'yes':1,'no':0})
raw_df['basement']=raw_df['basement'].map({'yes':1,'no':0})
raw_df['hotwaterheating']=raw_df['hotwaterheating'].map({'yes':1,'no':0})
raw_df['airconditioning']=raw_df['airconditioning'].map({'yes':1,'no':0})
# raw_df['parking']=raw_df['parking'].map({'yes':1,'no':0})
raw_df['prefarea']=raw_df['prefarea'].map({'yes':1,'no':0})
raw_df['furnishingstatus']=raw_df['furnishingstatus'].map({'furnished':2,'unfurnished':0,'semi-furnished':1})

input_cols = list(raw_df.columns)[1:]
target_col = 'price'

scaler = MinMaxScaler()
scaler.fit(raw_df[input_cols])

model = LinearRegression()
model.fit(raw_df[input_cols],raw_df[target_col])

# new_input = {'area':7420,
#              'bedrooms':4,
#              'bathrooms':2,
#              'stories':3,
#              'mainroad':1,
#              'guestroom':1,
#              'basement':1,
#              'hotwaterheating':1,
#              'airconditioning':1,
#              'parking':1,
#              'prefarea':1,
#              'furnishingstatus':2
# }

# input_data = pd.DataFrame([new_input])
# y = model.predict(input_data)
# print(y)
# print(raw_df['parking'])

pickle.dump(model,open('priceModel.pkl','wb'))