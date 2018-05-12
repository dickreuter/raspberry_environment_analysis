'''
Functions that are used to log and analyse present and past bets
'''

from datetime import datetime, date, time, timedelta

import pandas as pd
from pymongo import MongoClient


class MongoManager(object):
    def __init__(self):
        self.mongoclient = MongoClient('mongodb://rasp:rasp@dickreuter.com:27017/raspberry')
        self.mongodb = self.mongoclient['raspberry']

    def get_temp_for_plotting(self, max_rows=0):
        start_today = datetime.combine(date.today(), time(8))
        end_today = datetime.combine(date.today(), time(20))
        start_yesterday = datetime.combine(date.today() - timedelta(1), time(8))
        end_yesterday = datetime.combine(date.today() - timedelta(1), time(20))
        input_data = self.mongodb.temperatures
        data_today = pd.DataFrame(list(input_data.
                                       find({'timestamp': {'$gt': start_today, '$lt': end_today}}).
                                       limit(max_rows)))
        data_yesterday = pd.DataFrame(list(input_data.
                                           find({'timestamp': {'$gt': start_yesterday, '$lt': end_yesterday}}).
                                           limit(max_rows)))

        data_today = data_today[data_today['temperature'] > 0][['timestamp', 'temperature', 'port']]
        data_yesterday = data_yesterday[data_yesterday['temperature'] > 0][['timestamp', 'temperature', 'port']]

        data_today['timestamp'] = pd.to_datetime(data_today['timestamp'], format='%H:%M').dt.time
        data_yesterday['timestamp'] = pd.to_datetime(data_yesterday['timestamp'], format='%H:%M').dt.time
        return data_today, data_yesterday

    def get_temp_for_prediction(self, look_back, port):
        data = self.mongodb.temperatures.aggregate(
            [{'$match': {'port': port}},

             {'$sort': {'timestamp': -1}},
             {'$limit': look_back},
             {'$sort': {'timestamp': 1}},
             {'$project': {'temperature': 1}},

             ])
        df = pd.DataFrame(list(data))

        return df['temperature'].values.reshape(-1, 1)
