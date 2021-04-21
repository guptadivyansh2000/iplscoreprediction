# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 21:50:30 2021

@author: gupta
"""

import pickle

import numpy as np
from flask import Flask
# Importing essential libraries
from pywebio.input import *
from pywebio.platform.flask import webio_view

# Load the Random Forest CLassifier model
filename = 'first-innings-score-rf-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

'''@app.route('/')
def home():
    return render_template('index.html')
'''


@app.route('/')
def predict():
    temp_array = list()
    batting_team = input('batting-team')
    if batting_team == 'Chennai Super Kings':
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Delhi Daredevils':
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Kings XI Punjab':
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif batting_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif batting_team == 'Mumbai Indians':
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif batting_team == 'Rajasthan Royals':
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif batting_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif batting_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

    bowling_team = input('bowling-team')
    if bowling_team == 'Chennai Super Kings':
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Delhi Daredevils':
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kings XI Punjab':
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif bowling_team == 'Mumbai Indians':
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif bowling_team == 'Rajasthan Royals':
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif bowling_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif bowling_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

        overs = input('overs:', type=FLOAT)
        runs = input('runs', type=NUMBER)
        wickets = input('wickets', type=NUMBER)
        runs_in_prev_5 = input('runs_in_prev_5', type=NUMBER)
        wickets_in_prev_5 = input('wickets_in_prev_5', type=NUMBER)
        venue = input('venue')
        temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5, venue]

        data = np.array([temp_array])
        int(regressor.predict(data)[0])


app.add_url_rule('/tool', 'webio_view', webio_view(predict), methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    app.run(host='Localhost', port=80)
