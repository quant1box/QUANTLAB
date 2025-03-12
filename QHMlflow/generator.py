
import sys
sys.path.append('..')

import pymongo
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from typing import Dict, List, Union, Tuple
from QHUtil.tools import hot_map, pInfo, calc_trade_date

class SignalsGenerator(TransformerMixin):
    """信号生成"""

    def __init__(self, model, threshold:List=[], last_date: bool = False) -> None:
        super(SignalsGenerator, self).__init__()
        self.model = model
        self.threshold = threshold
        self.last_date = last_date

    def fit(self, X):
        return self

    def transform(self, X) -> pd.DataFrame:
        """"""
        df = X

        # 预测数据
        predict_data = pd.DataFrame(
            index=df.index, data=self.model.predict(df.values), columns=['sign'])


        buy_signal = predict_data.loc[predict_data.values >=
                                      self.threshold[1], 'sign']
        sell_signal = predict_data.loc[predict_data.values <=
                                       self.threshold[0], 'sign']

        signals = pd.concat([np.sign(buy_signal).astype(
            int), np.sign(sell_signal).astype(int)], axis=0).sort_index()

        if self.last_date:
            signals = signals.loc[signals.index.get_level_values(
                'date') == signals.index.get_level_values('date').max()]

        # 返回预测值
        return signals.reset_index()


class PositionGenerator(TransformerMixin):
    """calculate target positions"""

    def __init__(self,
                 price_data: pd.DataFrame,
                 total_portfolio_value: float = 1000000,
                 risk_per_trade: float = 0.05,
                 max_portfolio_risk: float = 0.10) -> None:

        self.price_data = price_data
        self.total_portfolio_value = total_portfolio_value
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk

    def calculate_position_size(self,
                                symbol: str,
                                current_price: float,
                                margin_rate: float = 0.13,
                                tick_size: float = 10) -> int:

        risk_per_trade_value = self.total_portfolio_value * self.risk_per_trade

        # 处理可能的异常
        try:
            max_position_size = (self.total_portfolio_value *
                                 self.max_portfolio_risk) / (current_price * margin_rate)
        except ZeroDivisionError:
            raise ValueError("当前价格为零，无法计算仓位大小。")

        # 计算仓位规模
        position_size = min(risk_per_trade_value /
                            (current_price * margin_rate), max_position_size)

        # 调整仓位规模
        position_size_adjusted = round(position_size / tick_size)

        return int(position_size_adjusted)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        # 确保 self.price_data 是有效的 DataFrame
        if not isinstance(self.price_data, pd.DataFrame) or self.price_data.empty:
            raise ValueError("无效的价格序列数据。")

        # 合并数据
        df = pd.merge(self.price_data['close'], X, on=['date', 'code'])
        df.reset_index(inplace=True)

        # 生成目标持仓
        return pd.DataFrame({
            'trade_date': calc_trade_date(df['date'].at[0]),
            'code': df['code'].map(lambda x: hot_map(x[:-2])),
            'share': df.apply(lambda x: self.calculate_position_size(
                symbol=x['code'],
                current_price=x['close'],
                tick_size=pInfo(x['code'][:-2])['volscale']
            ), axis=1)*df.sign
        })


class MongoPositionStorage(TransformerMixin):
    """MongoDB Target position storage"""

    def __init__(self, host: str = '192.168.31.220',
                 port: int = 27017,
                 db_name: str = 'quantaxis',
                 collection_name: str = None,
                 filters: Union[List, Tuple] = ['trade_date', 'code']) -> None:

        super(MongoPositionStorage, self).__init__()

        client = pymongo.MongoClient(host=host, port=port)
        data_base = client[db_name]
        self.collection = data_base[collection_name]
        self.filters = filters

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        df = X.copy()
        dt = df.to_dict(orient='records')

        if isinstance(dt, list):
            try:
                bulk_operations = [pymongo.UpdateOne(
                    {x: d[x] for x in self.filters},
                    {'$set': d},
                    upsert=True
                ) for d in dt]

                self.collection.bulk_write(bulk_operations)
            except Exception as e:
                print(f"写入数据库时发生错误: {e}")

        else:
            print('输入的数据格式不正确，请检查...')

        return X
