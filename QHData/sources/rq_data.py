# %%
from typing import Dict
import rqdatac
from pandas import DataFrame
import pymongo
import pandas as pd
import logging
import multiprocessing
# pool = multiprocessing.Pool(processes=4)


class rqdata:

    def __init__(self) -> None:

        # # 商品类型（88为主力连续）
        # self.stype = stype

        # 初始化数据库连接
        self.client = pymongo.MongoClient(host='192.168.31.220', port=27017)
        self.db = self.client['rq_data']

        # 在MongoDB中创建索引提高写入速度
        self.db['future_tick'].create_index([('symbol', 1), ('datetime', 1)])
        self.db['future_min'].create_index([('symbol', 1), ('datetime', 1)])
        self.db['future_day'].create_index([('symbol', 1), ('datetime', 1)])

        # 米框连接初始化
        rqdatac.init('license', 'S7W-IUE9eztHTQB396qfNBzXspIMGWnGRKQFRm-Tu9hlhw9uP26Ik40FNsPInfDQ3H574bLfDRw2MlBaavOU7ynwo3QLCR_7gRSwwWCdGsM8rqmsUd9SNqOI92vtEK8pwf-EEe6ob8l_JrFqts7q0jIZX3r3_eik10GcT0pirtY=fCfn-y4d_neLmJsYh9ql5BdzjPpeo1NGo3zqyVFeoth_ogUnkGe0KBUzH_UNUhdKVfLvIIP7wIM5tNSUnpqFxtXHXHmhpdR0Novtmg519Pd63vs995mDTnGEWEdp5CTdh0GWkOJWlCnnzBSi7LFVNzJyv_UQcwWIyh13sIMl7iI=')

        self.symbols = {
            "SHFE": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG", "RB", "WR", "HC", "SS", "BU", "RU", "NR", "SP", "SC", "LU", "FU"],
            "DCE": ["C", "CS", "A", "B", "M", "Y", "P", "FB", "BB", "JD", "RR", "L", "V", "PP", "J", "JM", "I", "EG", "EB", "PG"],
            "CZCE": ["SR", "CF", "CY", "PM", "WH", "RI", "LR", "AP", "JR", "OI", "RS", "RM", "TA", "MA", "FG", "SF", "ZC", "SM", "UR", "SA", "CJ"],
            "CFFEX": ["IH", "IC", "IF", "TF", "T", "TS"]
        }

        # 商品：交易所
        self.map = {m: k for k, v in self.symbols.items() for m in v}

    def query_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1m') -> DataFrame:
        """
        查询商品的历史数据
        :symbol 商品名称
        :start_date 开始日期
        :end_date   结束日期
        interval    时间周期（默认为1分钟）
        """

        df = rqdatac.get_price(symbol, start_date=start_date,
                               end_date=end_date, frequency=interval).reset_index()
        
        df['interval'] = interval

        df.rename(columns={'order_book_id': 'symbol',
                           'total_turnover': 'turnover'}, inplace=True)

        return df

    def save2mongo(self, df: pd.DataFrame) -> None:
        """save to mongodb"""
        # 保存逻辑
        self.db['future_min'].insert_many(df.to_dict('records'))

    def save2csv(self, df: pd.DataFrame) -> None:
        """
        save csv to local
        """
        cols = ['date', 'time', 'open', 'high', 'low',
                'close', 'turnover', 'open_interest']

        df.rename(columns={'trading_date': 'date'}, inplace=True)
        df['time'] = df['datetime'].apply(lambda x: str(x).split(' ')[1])

        result = df[cols]
        # -----------------------------------------
        product = df['symbol'].unique()[0][:-2]
        exchg = self.map.get(product)
        chars = list(df['interval'].unique()[0])
        interval = ''.join([chars[1], chars[0]])

        if exchg in ['SHFE','DCE']:
            file_name = exchg + '.' + product.lower() + '.' + 'HOT' + '_' + interval + '.csv'
        else:
            file_name = exchg + '.' + product + '.' + 'HOT' + '_' + interval + '.csv'


        result.to_csv(file_name, index=False)


# %%
if __name__ == "__main__":

    rq = rqdata()
    # --------------------
    start_date = '20231001'
    end_date = '20231020'
    # --------------------

    stype = '88'  # 合约类型为主力连续

    for code in rq.map.keys():
        try:
            print('Start querying %s', code)
            df = rq.query_data(symbol=code+stype, start_date=start_date, end_date=end_date, interval='1m')
            rq.save2mongo(df)
            # rq.save2csv(df)
            print('Saving %s data to MongoDB', code)
        except AttributeError as e:
            print(e)
