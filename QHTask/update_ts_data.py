

# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import logging
from QHUtil.loggers import watch_dog
from QHUtil.mongostorager import MongoStorager
from sklearn.pipeline import Pipeline
from datetime import datetime
from QUANTHUB.QHData.sources.ts_data import ts_fut_daily, ts_fut_holding
from QHUtil.data_base import tab_name, db_setting


# from data.ppi import get_basis, get_spot_future
# from data.fut_holding import get_fut_holding
# from data.kiiik import (
#     getLongShortInQuoteData,
#     getExchangeHoldPositions,
#     getLongShortInOrDecreaseData,
#     getLongShortRateData,
# )


host = db_setting.DB_NAS_HOST.value
db_name = db_setting.DB_NAME.value

logger = watch_dog(filename='storager.log', level=logging.DEBUG)


curr_date = datetime.now().strftime('%Y%m%d')


def update_fut_daily():

    table_name = tab_name.TS_FUT_DAILY.value

    data = ts_fut_daily(trade_date=curr_date)
    p = Pipeline([
        ('save', MongoStorager(table_name=table_name,
         filters=['trade_date', 'ts_code']))
    ])

    p.transform(data)

    logger.info(f'tushare 期货日线数据保存至: {host} > {db_name} > {table_name}')


def update_fut_holding():

    table_name = tab_name.TS_FUT_HOLDING.value

    data = ts_fut_holding(trade_date=curr_date)
    p = Pipeline([
        ('save', MongoStorager(table_name=table_name,
         filters=['trade_date', 'symbol', 'broker']))
    ])

    p.transform(data)

    logger.info(f'tushare 期货会员持仓数据保存至: {host} > {db_name} > {table_name}')

# %%

# if __name__ == '__main__':

#     import pandas as pd
#     date_range = [x.strftime('%Y%m%d') for x in pd.date_range(
#         start='20100101', end='20230101', freq='B')]

#     table_name = tab_name.TS_FUT_DAILY.value

#     for curr_date in date_range:
#         data = ts_fut_daily(trade_date=curr_date)
#         p = Pipeline([
#             ('save', MongoStorager(table_name=table_name,
#                                    filters=['trade_date','ts_code']))
#         ])

#         p.transform(data)

#         logger.info(
#             f'{[curr_date]} tushare 期货交易数据数据保存至: {host} > {db_name} > {table_name}')
