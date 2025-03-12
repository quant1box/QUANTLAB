import os
import json
import time
import requests
import pandas as pd
from pandas import DataFrame
import polars as pl
from QHData.query import Query
from typing import List, Union, Tuple
from datetime import datetime, timedelta
from sklearn.base import TransformerMixin
from QHFactor.transfromer import ReturnsTransformer


class ConfigLoader:

    def __init__(self) -> None:
        pass

    def load_sw_level1(self):
        """获取申万行业一级"""
        pass

    def load_sw_level2(self):
        """获取申万行业二级"""
        pass


class CodesLoader:

    def __init__(self, n: int = 20):
        self.n = n
        self.q = Query()

    def load_fut_codes(self, start_date=None, end_date=None):
        """加载期货代码"""
        try:
            if start_date is None or end_date is None:
                current_date = datetime.now()
                start_date = current_date - timedelta(days=365)
                start = start_date.strftime('%Y-%m-%d')
                end = current_date.strftime('%Y-%m-%d')
            else:
                start = start_date
                end = end_date

            q = Query()
            codes = q.fetch_future_codes(
                start_date=start,
                end_date=end,
                n=self.n
            )
            return codes
        except Exception as e:
            print(f"Error loading future codes: {e}")
            return None

    def load_stk_codes(self, category: str = 'zz500.json'):
        """
        加载股票代码列表

        Parameters:
        -----------
        category : str, default 'zz500.json'
            股票代码分类文件名，例如 'zz500.json', 'hs300.json' 等

        Returns:
        --------
        List[str]
            股票代码列表

        Raises:
        -------
        FileNotFoundError
            当指定的文件不存在时
        ValueError
            当文件格式不正确或category参数无效时
        """
        try:
            # 构建配置文件的完整路径
            config_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'QHConfig',
                category
            )

            # 验证文件路径
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Config file not found: {config_path}")

            # 验证文件扩展名
            if not category.endswith('.json'):
                raise ValueError("File must be a JSON file")

            # 读取并解析JSON文件
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")

            # 验证数据不为空
            if not data:
                raise ValueError("Empty data in JSON file")

            # 处理股票代码
            stock_codes = []
            for code in data:
                try:
                    # 假设代码格式为 "xxx.yyy"
                    stock_code = code.split('.')[0]
                    if stock_code:  # 确保提取的代码不为空
                        stock_codes.append(stock_code)
                except (IndexError, AttributeError):
                    print(f"Warning: Invalid code format: {code}")
                    continue

            # 验证是否成功提取到股票代码
            if not stock_codes:
                raise ValueError("No valid stock codes found in file")

            return stock_codes

        except Exception as e:
            print(f"Error loading stock codes: {e}")
            return None


class DataLoader(TransformerMixin):
    """data loader"""

    def __init__(self, start_date=None, end_date=None, freq: str = 'd'):
        """
        初始化 DataLoader

        Parameters:
        -----------
        start_date : str, optional
            数据开始日期，格式：'YYYY-MM-DD'
        end_date : str, optional
            数据结束日期，格式：'YYYY-MM-DD'
        freq : str, default 'd'
            数据频率，支持 'd'(日), 'w'(周), 'm'(月)等
        """
        super().__init__()

        # 验证日期格式
        if start_date and not self._is_valid_date(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}")
        if end_date and not self._is_valid_date(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}")

        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.q = Query()

    @staticmethod
    def _is_valid_date(date_str):
        """验证日期格式"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def load_future(self, codes: List[str] = None, return_X_y: bool = False) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        """
        加载期货数据

        Parameters:
        -----------
        codes : List[str], optional
            期货代码列表
        return_X_y : bool, default False
            是否返回特征和标签

        Returns:
        --------
        Union[DataFrame, Tuple[DataFrame, DataFrame]]
            根据 return_X_y 返回数据或数据和标签的元组
        """
        try:
            if not codes:
                codes = CodesLoader(n=30).load_fut_codes()
                if not codes:
                    raise ValueError("Failed to load future codes")

            data = self.q.fetch_future_data(
                codes=codes,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=self.freq
            )

            if data.empty:
                raise ValueError(
                    "No data retrieved for the specified parameters")

            if return_X_y:
                returns = ReturnsTransformer(period=1).transform(data)
                return data, returns

            return data

        except Exception as e:
            print(f"Error loading future data: {e}")
            return None

    def load_h5(self, file_path: str):
        """加载本地hdf5文件"""
        pass

    def load_parquet(self, file_path: str, return_X_y: bool = False) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        """
        加载本地 parquet 文件

        Parameters:
        -----------
        file_path : str
            parquet 文件路径
        return_X_y : bool, default False
            是否返回特征和标签

        Returns:
        --------
        Union[DataFrame, Tuple[DataFrame, DataFrame]]
            根据 return_X_y 返回数据或数据和标签的元组
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            df = pd.read_parquet(file_path)

            if df.empty:
                raise ValueError("Empty dataframe loaded from parquet file")

            if return_X_y:
                returns = LogReturnsTransformer(period=1).transform(df)
                return df, returns

            return df

        except Exception as e:
            print(f"Error loading parquet file: {e}")
            return None

    def load_stock(self, codes: List = [], return_X_y: bool = False) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        """
        加载股票数据

        Parameters:
        -----------
        codes : List[str]
            股票代码列表
        return_X_y : bool, default False
            是否返回特征和标签

        Returns:
        --------
        Union[DataFrame, Tuple[DataFrame, DataFrame]]
            根据 return_X_y 返回数据或数据和标签的元组
        """
        try:
            if not codes:
                raise ValueError("Stock codes list cannot be empty")

            stock_daily = self.q.fetch_stock_data(
                codes=codes,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=self.freq
            )

            data = stock_daily.drop_duplicates().dropna()

            # 验证数据
            if data.empty:
                raise ValueError(
                    "No data retrieved for the specified parameters")

            # 返回数据
            if return_X_y:
                returns = ReturnsTransformer(period=1).transform(data)
                return data, returns

            return data

        except Exception as e:
            print(f"Error loading stock data: {e}")
            return None

    def load_bench_market(self, code: str, return_X_y: bool = False) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        """加载市场基准数据并进行重采样

        Parameters:
        -----------
        code : str
            市场基准代码，例如 '000300.SH' 表示沪深300
        freq : str, default 'd'
            重采样频率，支持 'd'（日）、'w'（周）、'm'（月）

        Returns:
        --------
        DataFrame
            市场基准数据
        """
        try:
            if not code:
                raise ValueError("Market benchmark code cannot be empty")

            benchmark_data = self.q.fetch_bench_market_data(
                codes=code,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=self.freq
            )

            data = benchmark_data.drop_duplicates().dropna()  # 获取数据部分

            # 验证数据
            if data.empty:
                raise ValueError(
                    "No data retrieved for the specified parameters")

            # 返回数据
            if return_X_y:
                returns = ReturnsTransformer(period=1).transform(data)
                return data, returns

            return data

        except Exception as e:
            print(f"Error loading benchmark market data: {e}")
            return None

    def load_crypto(self, symbol: str, limit: int = 1000) -> DataFrame:
        """
        从币安获取K线数据

        Parameters:
        -----------
        symbol : str
            交易对，例如 'BTCUSDT'
        limit : int, default 1000
            每次请求的数据条数，最大1000

        Returns:
        --------
        DataFrame
            K线数据
        """
        try:
            if not symbol:
                raise ValueError("Symbol cannot be empty")

            api_key = 'YOUR_API_KEY'  # 建议使用环境变量存储
            url = 'https://api.binance.com/api/v3/klines'
            headers = {'X-MBX-APIKEY': api_key}
            all_klines = []

            # 时间转换
            start_date = int(datetime.strptime(
                self.start_date, '%Y-%m-%d').timestamp() * 1000) if self.start_date else None
            end_date = int(datetime.strptime(
                self.end_date, '%Y-%m-%d').timestamp() * 1000) if self.end_date else None

            while True:
                params = {
                    'symbol': symbol,
                    'interval': self.freq,
                    'limit': limit
                }
                if start_date:
                    params['startTime'] = start_date
                if end_date:
                    params['endTime'] = end_date

                response = requests.get(url, params=params, headers=headers)
                data = response.json()

                if "code" in data:
                    raise Exception(f"Binance API error: {data['msg']}")

                if not data:
                    break

                all_klines.extend(data)

                if len(data) < limit:
                    break

                start_date = data[-1][6] + 1
                time.sleep(1)  # API 限流

            if not all_klines:
                raise ValueError("No data retrieved from Binance API")

            # 数据转换
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # 数据处理
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.rename(columns={'timestamp': 'datetime'}, inplace=True)
            df['code'] = symbol
            df.set_index(['datetime', 'code'], inplace=True)

            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                               'quote_asset_volume', 'taker_buy_base_asset_volume',
                               'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            return df

        except Exception as e:
            print(f"Error loading crypto data: {e}")
            return None
