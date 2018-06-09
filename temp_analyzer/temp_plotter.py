from sys import platform

import matplotlib

if platform == "linux" or platform == "linux2":
    matplotlib.use('Agg')
import pandas as pd

from temp_analyzer.mongo_manager import MongoManager


def get_temp(min_limit, max_limit):
    m = MongoManager()
    df_today, df_yesterday = m.get_temp_for_plotting()
    df_today['day'] = 'today'
    df_yesterday['day'] = 'yesterday'
    df = pd.concat([df_today, df_yesterday])
    df['timestamp'] = df['timestamp'].astype(str).str[0:2]
    df = df.set_index(['timestamp', 'port', 'day'])
    df = df.unstack(-1).unstack(-1)
    df.columns = ['today sensor 1', 'today sensor 2', 'yesterday sensor 1', 'yesterday sensor 2']
    df = df.reset_index()
    df = df.rename(columns={"timestamp": "hour"})
    ax = df.plot(xticks=df.index, style=['bs-', 'go-', 'b:', 'g:'])
    ax.set_xticklabels(df["hour"]);
    ax.set_title('Temperatures 6th Floor 5NC')
    ax.set_ylabel('Celsius')
    ax.set_xlabel('Time')
    ax.axhline(y=max_limit, color='r', linestyle='-')
    ax.axhline(y=min_limit, color='b', linestyle='-')

    ax.get_figure().savefig('chart.jpg')
    return df


if __name__ == '__main__':
    get_temp(21.5, 28)
