import polars as pl
import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from sklearn.base import TransformerMixin

from QHFactor.fn import *
from QHFactor.factors import GTJA_191
from QHFactor.factors_map import gtja_191_map


class PolarsFactorCalculator(TransformerMixin):
    """使用Polars实现的因子计算器,优化计算效率"""
    
    def __init__(self,
                 factors: Union[List[str], Tuple[str]] = None,
                 pre_process: Optional[Callable] = None,
                 post_process: Optional[Callable] = None) -> None:
        """
        初始化因子计算器
        
        Args:
            factors: 要计算的因子列表
            pre_process: 数据预处理函数
            post_process: 数据后处理函数
        """
        # if factors:
            # 标准化因子名称（移除下划线）
            # self.factors = [f.replace('_', '') for f in factors]
            # 检查因子名称是否存在
            # invalid_factors = [f for f in self.factors if f not in gtja_191_map]
        #     # if invalid_factors:
        #     #     raise ValueError(f"Invalid factor names: {invalid_factors}. "
        #     #                    f"Available factors: {list(gtja_191_map.keys())}")
        # else:
        #     self.factors = list(gtja_191_map.keys())
        
        self.pre_process = pre_process
        self.post_process = post_process

    def _process_single_stock(self, stock_data: pl.DataFrame) -> pl.DataFrame:
        """处理单个股票的因子计算"""
        try:
            # 确保数据按时间排序
            stock_data = stock_data.sort('datetime')
            
            # 使用polars优化的方式获取数据
            ohlcv_data = (stock_data
                         .select(['open', 'high', 'low', 'close', 'volume'])
                         .to_numpy())

            # 检查数据是否为空
            if len(ohlcv_data) == 0:
                raise ValueError("Empty data")

            # 计算因子
            gtja = GTJA_191(ohlcv_data)
            factors_values = {}

            for factor in self.factors:
                try:
                    factor_value = gtja_191_map[factor](gtja)
                    factors_values[factor] = factor_value
                except Exception as e:
                    print(f"Failed to calculate {factor} for {stock_data['code'][0]}: {str(e)}")
                    factors_values[factor] = np.zeros(len(ohlcv_data))

            # 构建结果DataFrame
            result = pl.DataFrame({
                'datetime': stock_data['datetime'],
                'code': stock_data['code'],
                **{f: pl.Series(factors_values[f], dtype=pl.Float32) for f in self.factors}
            })

            # 替换无穷值
            result = result.with_columns([
                pl.exclude(['datetime', 'code'])
                # .map(lambda x: x.replace({float('inf'): np.nan, float('-inf'): np.nan}))
                # .fill_null(0)
            ]).fill_nan(0).fill_null(0)

            return result
            
        except Exception as e:
            print(f"Error processing stock {stock_data['code'][0]}: {str(e)}")
            return pl.DataFrame()  # 返回空DataFrame而不是直接失败

    def transform(self, X) -> pl.DataFrame:
        """转换数据并计算因子"""
        try:
            # 检查必需的列
            required_columns = ['datetime', 'code', 'open', 'high', 'low', 'close', 'volume']
            if isinstance(X, pl.DataFrame):
                missing_cols = [col for col in required_columns if col not in X.columns]
            else:
                missing_cols = [col for col in required_columns if col not in X.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 转换为Polars DataFrame
            if not isinstance(X, pl.DataFrame):
                df = pl.from_pandas(X)
            else:
                df = X

            # 检查数据是否为空
            if len(df) == 0:
                raise ValueError("Empty input data")

            # 优化数据类型
            df = df.with_columns([
                pl.col('code').cast(pl.Categorical),
                pl.col('^(open|high|low|close)$').cast(pl.Float32),
                pl.col('volume').cast(pl.Float32)
            ])

            # 使用polars的groupby操作
            results = (df.group_by('code')
                      .map_groups(self._process_single_stock))

            if len(results) == 0:
                raise ValueError("No factors were calculated successfully")

            # 优化排序和索引设置
            final_df = (results
                       .sort(['datetime', 'code'])
                       )

            return final_df

        except Exception as e:
            raise RuntimeError(f"Factor calculation failed: {str(e)}")

    def fit(self, X, y=None):
        return self

