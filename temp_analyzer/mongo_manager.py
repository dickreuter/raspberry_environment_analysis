'''
Functions that are used to log and analyse present and past bets
'''

from datetime import datetime, date, time, timedelta

import pandas as pd
from pymongo import MongoClient


class MongoManager(object):
    def __init__(self):
        self.mongoclient = MongoClient('mongodb://raspberry:raspberry@52.201.173.151:27017/admin')
        self.mongodb = self.mongoclient.raspberry

    def get_temp_df(self, max_rows=0):
        start_today = datetime.combine(date.today(), time(9))
        end_today = datetime.combine(date.today(), time(20))
        start_yesterday = datetime.combine(date.today() - timedelta(1), time(9))
        end_yesterday = datetime.combine(date.today() - timedelta(1), time(20))
        input_data = self.mongodb.temperatures
        data_today = pd.DataFrame(list(input_data.
                                 find({'timestamp': {'$gt': start_today, '$lt': end_today}}).
                                 limit(max_rows)))
        data_yesterday = pd.DataFrame(list(input_data.
                                 find({'timestamp': {'$gt': start_yesterday, '$lt': end_yesterday}}).
                                 limit(max_rows)))

        data_today = data_today[data_today['temperature'] > 0][['timestamp', 'temperature']]
        data_yesterday = data_yesterday[data_yesterday['temperature'] > 0][['timestamp', 'temperature']]

        data_today['timestamp'] = pd.to_datetime(data_today['timestamp'], format='%H:%M').dt.time
        data_yesterday['timestamp'] = pd.to_datetime(data_yesterday['timestamp'], format='%H:%M').dt.time
        return data_today,data_yesterday