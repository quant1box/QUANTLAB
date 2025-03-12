
# tushare 数据源

import tushare as ts
import pandas as pd
from time import sleep
from QHUtil.tools import chunks

pro = ts.pro_api('3ce4674d1ac49d9789c6c1353e7170789be05cd485aebbca74c9f51c')

# 期货交易所
exchangs = ['SHFE', 'DCE', 'CZCE', 'CFFEX']


def get_fut_basic(exchange: str):
    """
    期货合约列表
    """
    pass


def get_trade_cal():
    """
    期货交易日历
    """
    pass


def ts_fut_daily(trade_date: str):
    """
    期货日线数据
    """
    dataArr = []
    for x in exchangs:
        try:
            df = pro.fut_daily(trade_date=trade_date, exchange=x)
            df.fillna(value=0, inplace=True)
            dataArr.append(df)
            sleep(1.)

        except (ValueError):
            continue
    return pd.concat(dataArr, axis=0)


def ts_fut_holding(trade_date: str):
    """
    会员持仓数据(tushare)
    """

    dataArr = []

    for ex in exchangs:
        df = pro.fut_holding(trade_date=trade_date, exchange=ex)
        df.fillna(value=0, inplace=True)
        dataArr.append(df)
        sleep(1)

    return pd.concat(dataArr, axis=0)


def ts_fut_wsr(trade_date: str, symbol: str):
    """
    仓单日报
    """
    df = pro.fut_wsr(trade_date=trade_date, symbol=symbol)
    data = df.to_dict(orient='records')
    for d in data:
        yield d


def ts_fut_settle(trade_date: str):
    """
    结算参数
    """
    df = pro.fut_settle(trade_date=trade_date)
    data = df.to_dict(orient='records')
    for d in data:
        yield d
