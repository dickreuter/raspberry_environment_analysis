import datetime

import matplotlib.pyplot as plt
import pandas as pd

from temp_analyzer.mongo_manager import MongoManager


def get_temp():
    m = MongoManager()
    df, df2 = m.get_temp_df()
    df = df.set_index('timestamp')
    x = [(i, datetime.time(i)) for i in range(8, 20)]

    df3 = pd.merge(df2.reset_index(), df.reset_index(), on='timestamp', how='outer')
    df3 = df3.set_index('timestamp')
    df3 = df3.drop('index', 1)
    df3.plot()
    plt.show()


if __name__ == '__main__':
    get_temp()
