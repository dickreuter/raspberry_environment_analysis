import datetime

import matplotlib.pyplot as plt
import pandas as pd

from temp_analyzer.mongo_manager import MongoManager


def get_temp():
    m = MongoManager()
    df, df2 = m.get_temp_df()
    df.index = df['timestamp']
    df2.index = df2['timestamp']
    del df['timestamp'], df2['timestamp']
    df.columns = ['Temperature1']
    df2.columns = ['Temperature2']
    ax = df.plot()
    df2.plot(ax=ax)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('temp_charts.jpg')
    plt.show()

if __name__ == '__main__':
    get_temp()
