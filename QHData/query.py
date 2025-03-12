
import pymongo
import datetime
import pandas as pd
import polars as pl
from QHData.config import Config
from typing import Union, List, Tuple


class Query:
    """数据查询类,用于从MongoDB数据库中获取和处理金融市场数据"""

    def __init__(self):
        """初始化查询类,建立MongoDB连接"""
        self.client = pymongo.MongoClient(**Config.MONGODB)
        self.db = self.client['quantaxis']
    #----------------`重采样`--------------------------------
    def resample_data(self, df: pl.DataFrame, frequency: str) -> pl.DataFrame:
        """将日度数据重采样为周度或月度数据

        Args:
            df: 输入的DataFrame,包含日度数据
            frequency: 重采样频率,'weekly'或'monthly'

        Returns:
            重采样后的DataFrame,包含OHLCV数据

        Raises:
            ValueError: 当frequency不是'weekly'或'monthly'时抛出
        """
        if frequency not in ['weekly', 'monthly']:
            raise ValueError("Frequency must be 'weekly' or 'monthly'.")

        # 根据频率进行重采样
        freq_map = {
            'weekly': '1w',
            'monthly': '1mo',
            'quarterly': '1q',
            'yearly': '1y'
        }

        # 对每个code分组后再按时间重采样
        resampled_df = (
            df.group_by(
                ['code', pl.col('date').dt.truncate(freq_map[frequency])])
            .agg(
                pl.first('open').alias('open'),
                pl.max('high').alias('high'),
                pl.min('low').alias('low'),
                pl.last('close').alias('close'),
                pl.sum('volume').alias('volume')
            ).sort(['date', 'code']))  # 确保结果按code和日期排序

        return resampled_df

    def resample_data_pd(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """使用pandas将日度数据重采样为周度或月度数据

        Args:
            df: 输入的DataFrame,包含日度数据
            frequency: 重采样频率,'weekly'或'monthly'

        Returns:
            重采样后的DataFrame,包含OHLCV数据

        Raises:
            ValueError: 当frequency不是'weekly'或'monthly'时抛出
        """
        if frequency not in ['weekly', 'monthly','quarterly','yearly']:
            raise ValueError("Frequency must be 'weekly' or 'monthly' or 'quarterly' or yearly.")


        # 根据频率进行重采样
        freq_map = {
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'yearly': 'Y'
        }

        # 对每个code分组后再按时间重采样
        rsp_dict = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        result = df.groupby(level=1).apply(lambda x:x.groupby(pd.Grouper(level=0,freq=freq_map[frequency])).agg(rsp_dict)).swaplevel(0,1).sort_index()

        return result

    #----------------`期货`--------------------------------
    def fetch_future_list(self) -> pl.DataFrame:
        """获取期货合约列表

        Returns:
            包含期货代码和名称的DataFrame
        """
        results = list(self.db['future_list'].find(
            {}, {'_id': 0, 'code': 1, 'name': 1}))

        df = pl.DataFrame(results).filter(pl.col('code').str.contains('L8'))
        return df

    def fetch_future_codes(self,start_date: str, end_date: str = None, n: int = 30) -> List[str]:
        """获取期货代码

        Args:
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天
            n: 数量

        Returns:
            包含期货代码的列表
         # 构建查询。
        collection = 'future_day'
        """

        pipeline = [
            {
                "$match": {
                    "date": {"$gte": start_date, "$lte": end_date},
                    "code": {"$regex": "^.*L8.*$"}
                }
            },
            {
                "$group": {
                    "_id": "$code",
                    "pos_avg": {"$avg": "$position"},
                    "trade_avg": {"$avg": "$trade"},
                }
            },
            {
                "$sort": {
                    "trade_avg": -1,
                    "pos_avg": -1
                }
            },
            {
                "$limit": n
            }
        ]

        # 执行查询。
        results = self.db['future_day'].aggregate(pipeline)

        # 获取股票代码。
        codes = []
        for result in results:
            codes.append(result["_id"])

        return codes
    
    def fetch_future_data(self, codes: Union[str, List[str]], start_date: str, end_date: str = None, frequency: str = 'd') -> pd.DataFrame:
        """获取期货行情数据

        Args:
            codes: 期货代码，可以是单个字符串或字符串列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天
            frequency: 数据频率，支持:
                - 'd'/'D': 日线数据
                - 'weekly': 周线数据 
                - 'monthly': 月线数据
                - '60min','30min','15min','5min','1min': 分钟数据

        Returns:
            pl.DataFrame: 包含期货OHLCV数据的DataFrame，按日期和代码排序

        Raises:
            ValueError: 当codes为空或frequency不支持时抛出
        """
        # 参数验证和预处理
        if not codes:
            raise ValueError("codes cannot be empty")

        if end_date is None:
            end_date = datetime.date.today().strftime('%Y-%m-%d')

        codes_list = [codes] if isinstance(codes, str) else codes

        # 构造查询条件
        filters = {
            "code": {"$in": codes_list},
            "date": {
                "$gte": start_date,
                "$lte": end_date
            }
        }

        if frequency.lower() in ['d', 'daily', 'weekly', 'monthly']:

            # 日线数据处理
            # 使用索引优化查询
            day_dis = {'_id':0,'price':0,'amount':0,'position':0,'date_stamp':0}
            results = list(self.db['future_day'].find(filters,day_dis))  # 排除_id字段)

            cols = ['open','high','low','close']

            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'trade':'volume'},inplace=True)
            df.set_index(['date','code'],inplace=True)
            df[cols] = df[cols].transform(lambda x:round(x,2))


            # 根据频率处理数据
            if frequency.lower() in ['d', 'daily']:
                return df.drop_duplicates().sort_index()

            elif frequency in ['weekly', 'monthly']:
                return self.resample_data_pd(df,frequency=frequency).sort_index()

        elif frequency in ['60min', '30min', '15min', '5min', '1min']:

            # 构造查询条件
            filters = {
                "code": {"$in": codes},
                "date": {
                    "$gte": start_date,
                    "$lte": end_date
                },
                "type": frequency,
            }

            min_dis = {'_id': 0, 'date_stamp': 0,'time_stamp': 0, 'date': 0,'price':0,'amount':0,'position':0,'tradetime':0}

            # 执行查询
            results = list(self.db['future_min'].find(filters, min_dis))
            cols = ['open','high','low','close']
            # .ASCENDING), ('code', pymongo.ASCENDING)]))
            df = pd.DataFrame(results)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.rename(columns={'trade':'volume'},inplace=True)
            df.set_index(['datetime','code'],inplace=True)
            df[cols] = df[cols].transform(lambda x:round(x,2))

            return df.drop_duplicates().sort_index()
        else:
            raise ValueError(
                "Frequency must be 'd'/'D', 'weekly', 'monthly', '60min', '30min', '15min', '5min' or '1min'")

    #----------------`股票`--------------------------------
    def fetch_stock_list(self) -> pd.DataFrame:
        """获取股票列表

        Returns:
            包含股票代码和名称的DataFrame
        """
        results = list(self.db['stock_list'].find(
            {}, {'_id': 0, 'code': 1, 'name': 1}))

        df = pd.DataFrame(results)
        return df

    def fetch_stock_data(self, codes: Union[str, List[str]], start_date: str, end_date: str = None, frequency: str = 'd') -> pd.DataFrame:
        """获取股票行情数据

        Args:
            codes: 股票代码，可以是单个字符串或字符串列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天
            frequency: 数据频率，支持'd'(日),'weekly'(周),'monthly'(月)

        Returns:
            包含前复权后的股票OHLCV数据的DataFrame
        """
        if end_date is None:
            end_date = datetime.date.today().strftime('%Y-%m-%d')

        if isinstance(codes, str):
            codes = [codes]  # 确保codes是列表

        # 构造查询条件
        filters = {
            "code": {"$in": codes},
            "date": {
                "$gte": start_date,
                "$lte": end_date
            },
        }

        # 执行查询
        results = list(self.db['stock_day'].find(
            filters, {'_id': 0, 'date_stamp': 0}))

        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={'vol':'volume'},inplace=True)
        df.set_index(['date','code'],inplace=True)

        adj_factors = self.fetch_stock_adj(codes=codes, start_date=start_date, end_date=end_date)
        df = self.adj_stock_data(df, adj_factors, adj='qfq')

        if frequency == 'd':
            return df
        elif frequency in ['weekly', 'monthly','quarterly','yearly']:
            return self.resample_data_pd(df,frequency=frequency).sort_index()

    def fetch_stock_adj(self, codes: Union[str, List[str]], start_date: str, end_date: str = None) -> pd.DataFrame:
        """获取股票复权因子数据

        Args:
            codes: 股票代码，可以是单个字符串或字符串列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天

        Returns:
            包含复权因子的DataFrame
        """
        if end_date is None:
            end_date = datetime.date.today().strftime('Y%-%m-%d')
        if isinstance(codes, str):
            codes = [codes]  # 确保codes是列表

        # 构造查询条件
        filters = {
            "code": {"$in": codes},
            "date": {
                "$gte": start_date,
                "$lte": end_date
            },
        }

        # 执行查询
        results = list(self.db['stock_adj'].find(filters, {'_id': 0, 'date_stamp': 0}))

        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date','code'],inplace=True)

        return df.sort_index()

    def adj_stock_data(self, df: pd.DataFrame, adj_factor: pd.DataFrame, adj: str = 'qfq') -> pd.DataFrame:
        """对股票数据进行复权计算

        Args:
            df: 股票数据，包含 date、code、open、high、low、close、volume 列，index为['date','code']
            adj_factor: 复权因子数据，包含 adj 列，index为['date','code']
            adj: 复权方式，'qfq' 前复权，'hfq' 后复权

        Returns:
            复权后的股票数据DataFrame，index为['date','code']
        """
        # 合并数据，两个DataFrame的index都是['date','code']，可以直接join
        merged = df.join(adj_factor, how='left')

        # 需要复权的价格列
        price_cols = ['open', 'high', 'low', 'close']

        if adj == 'qfq':
            # 获取每个股票最新的复权因子
            latest_factors = adj_factor.groupby(level=1)['adj'].last()
            # 计算前复权价格
            for col in price_cols:
                merged[col] = merged[col] * merged['adj'] / merged.groupby(level=1)['adj'].transform('last')
        elif adj == 'hfq':
            # 获取每个股票最早的复权因子
            first_factors = adj_factor.groupby(level=1)['adj'].first()
            # 计算后复权价格
            for col in price_cols:
                merged[col] = merged[col] * merged['adj'] / merged.groupby(level=1)['adj'].transform('first')

        # 只保留需要的列，index保持不变
        result = merged[price_cols + ['volume']]
        result[price_cols] = result[price_cols].transform(lambda x:round(x,2))
        
        return result.sort_index()

    def fetch_bench_market_data(self, codes: Union[str, List[str]], start_date: str, end_date: str = None,frequency : str = 'd') -> pd.DataFrame:
        """获取指数行情数据

        Args:
            codes: 指数代码，可以是单个字符串或字符串列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天

        Returns:
            包含指数OHLCV数据的DataFrame
        """
        if end_date is None:
            end_date = datetime.date.today().strftime('%Y-%m-%d')
        if isinstance(codes, str):
            codes = [codes]  # 确保codes是列表

        # 构造查询条件
        filters = {
            "code": {"$in": codes},
            "date": {
                "$gte": start_date,
                "$lte": end_date
            },
        }

        dis = {'_id':0,'date_stamp':0,'up_count':0,'down_count':0,'amount':0}
        # 执行查询
        results = list(self.db['index_day'].find(filters, dis))

        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={'vol':'volume'},inplace=True)
        df.set_index(['date','code'],inplace=True)
        
        df = df.drop_duplicates().sort_index()

        if frequency  in ['d','daily']:
            return df
        elif frequency in ['weekly','monthly','quaterly','yearly']:
            return self.resample_data_pd(df,frequency).sort_index()
    

    def fetch_financial_data(self, codes: Union[str, List[str]], start_date: str, end_date: str = None) -> pl.DataFrame:
        """# TODO 获取财务数据

        Args:
            codes: 股票代码，可以是单个字符串或字符串列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天
        """
        

    
    #----------------`加密货币`--------------------------------
    def fetch_crypto_list(self) -> pl.DataFrame:
        """获取加密货币列表

        Returns:
            包含加密货币代码和名称的DataFrame
        """
        pass
    
    def fetch_crypto_codes(self,start_date: str, end_date: str = None, n: int = 30) -> List[str]:
        """获取加密货币代码

        Args:
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天
            n: 数量
        """
        pass

    def fetch_crypto_data(self, codes: Union[str, List[str]], start_date: str, end_date: str = None, frequency: str = 'd') -> pl.DataFrame:
        """获取加密货币行情数据

        Args:
            codes: 加密货币代码，可以是单个字符串或字符串列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天
            frequency: 数据频率，支持'd'(日),'weekly'(周),'monthly'(月)
        """
        pass

    #----------------`指数`--------------------------------
    def fetch_index_list(self) -> pl.DataFrame:
        """获取指数列表

        Returns:
            包含指数代码和名称的DataFrame
        """
        pass

    def fetch_index_data(self, codes: Union[str, List[str]], start_date: str, end_date: str = None, frequency: str = 'd') -> pl.DataFrame:
        """获取指数行情数据

        Args:
            codes: 指数代码，可以是单个字符串或字符串列表
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'，默认为当天
            frequency: 数据频率，支持'd'(日),'weekly'(周),'monthly'(月)
        """
        pass