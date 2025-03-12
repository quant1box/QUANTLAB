from datetime import datetime, date
from contextlib import closing
from tqsdk import TqApi, TqAuth, TqSim
from tqsdk.tools import DataDownloader
from tqsdk.tafunc import time_to_datetime
import json
import pandas as pd
import random
import time
import pymongo
from contextlib import closing

client = pymongo.MongoClient(host='127.0.0.1', port=27017)
database = client['tqdata']

api = TqApi(auth=TqAuth("17738702282", "cheung0927"))
download_tasks = {}


# 构建下载商品列表
path = '/Users/chiang/QUANTHUB/demos/test_hotpicker/hots.json'
with open(path, 'rb') as f:
    hot_map = json.load(f)
symbols = [''.join(['KQ.m@', k+'.'+m])
           for k, v in hot_map.items() for m in v.keys()]



for s in symbols[75:]:
    # 使用with closing机制确保回测完成后释放对应的资源
        print(s)
        try:
            df = api.get_kline_serial(
            symbol=s, duration_seconds=300, data_length=8000)

            # df['datetime'] = pd.to_datetime(df['datetime'])
            df['datetime'] = df['datetime'].apply(lambda x:time_to_datetime(x))
            df['symbol'] = df['symbol'].apply(lambda x: x.split('@')[1])
            df.drop(columns=['id'],inplace=True)

            data = df.to_dict(orient='records')
            for d in data:
                if database['future_min'].insert_one(d):
                    print(d)
            
        except Exception (KeyError,RuntimeError) as e:
            print(e)
            api.close()
            continue

        time.sleep(random.randint(1, 5))



    # ------------------------------------------------------    


# 使用with closing机制确保下载完成后释放对应的资源
# with closing(api):
#     while not all([v.is_finished() for v in download_tasks.values()]):
#         api.wait_update()
#         print("progress: ", {k: ("%.2f%%" % v.get_progress())
#               for k, v in download_tasks.items()})
