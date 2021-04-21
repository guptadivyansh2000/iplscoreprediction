#!/usr/bin/env python
# coding: utf-8
import pandas as pd

import pickle


matches_df=pd.read_csv('https://raw.githubusercontent.com/anujvyas/IPL-First-Innings-Score-Prediction-Deployment/master/ipl.csv')
matches_df.head(5)




df=matches_df.drop(['mid', 'batsman', 'bowler', 'striker', 'non-striker'],axis=1)





#df.head()

#df['bat_team'].unique()


df = df[df['overs']>=5.0]


consistent_teams=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab','Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']

df=df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]



#df['bat_team'].unique()
# convert date column datatype from string into date time object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
encod_df=pd.get_dummies(df,columns=['bat_team','bowl_team'])
#encod_df.head()
#encod_df.columns





encod_df=encod_df[['date','bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'total']]





X_train = encod_df.drop(labels='total', axis=1)[encod_df['date'].dt.year <= 2016]
X_test = encod_df.drop(labels='total', axis=1)[encod_df['date'].dt.year >= 2017]





y_train = encod_df[encod_df['date'].dt.year <= 2016]['total'].values
y_test = encod_df[encod_df['date'].dt.year >= 2017]['total'].values





X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)





'''from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
prediction=regressor.predict(X_test)
import seaborn as sns
sns.distplot(y_test-prediction)
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction))) '''



from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,max_depth=10)
rf.fit(X_train,y_train)
pred=rf.predict(X_test)
rf.score(X_train,y_train)


'''sns.distplot(y_test-pred)
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
'''

filename = 'first-innings-score-rf-model.pkl'
pickle.dump(rf, open(filename, 'wb'))

