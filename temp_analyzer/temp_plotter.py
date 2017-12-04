import matplotlib
matplotlib.use('Agg')
import pandas as pd



from temp_analyzer.mongo_manager import MongoManager


def get_temp():
    m = MongoManager()
    df_today, df_yesterday = m.get_temp_df()
    df_today['day'] = 'today'
    df_yesterday['day'] = 'yesterday'
    df = pd.concat([df_today, df_yesterday])
    df['timestamp'] = df['timestamp'].astype(str).str[0:2]
    df = df.set_index(['timestamp', 'port', 'day'])
    df = df.unstack(-1).unstack(-1)
    df.columns = ['today sensor 1', 'today sensor 2', 'yesterday sensor 1', 'yesterday sensor 2']
    fig = df.plot()
    fig.get_figure().savefig('chart.jpg')
    # plt.show()
    return df


if __name__ == '__main__':
    get_temp()
