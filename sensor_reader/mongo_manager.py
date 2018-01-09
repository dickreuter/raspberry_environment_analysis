'''
Functions that are used to log and analyse present and past bets
'''

from pymongo import MongoClient


class MongoManager(object):
    def __init__(self):
        self.mongoclient = MongoClient('mongodb://raspberry:raspberry@dickreuter.com:27017/admin')
        self.mongodb = self.mongoclient.raspberry

    def add_temp_reading(self, temp):
        self.mongodb['temperatures'].insert(temp)

    def get_daily_summary(self):
        res = self.mongodb['temperatures'].find()
        return res
