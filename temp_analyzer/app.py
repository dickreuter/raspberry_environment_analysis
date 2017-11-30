from temp_analyzer.mongo_manager import MongoManager
import matplotlib.pyplot as plt


def get_temp():
    m = MongoManager()
    df,df2 = m.get_temp_df()
    df=df.set_index('timestamp')
    ax=df.plot()
    df2.plot(ax=ax)
    plt.show()

if __name__ == '__main__':
    get_temp()
