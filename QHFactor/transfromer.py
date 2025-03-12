
import numpy as np
import polars as pl
import pandas as pd
from QHFactor.fn import *
from sklearn.base import TransformerMixin


# 滚动标准化
class RollingZScoreTransformer(TransformerMixin):
    """
    滚动标准化转换器类
    """

    def __init__(self, window=20):
        """
        初始化参数window为滚动窗口大小
        """
        self.window = window

    def fit(self, X, y=None):
        """
        fitting方法,直接返回self
        """
        return self

    def transform(self, X, y=None):
        """使用Polars计算移动zscore"""
        # 检查输入数据
        if X is None or (isinstance(X, pd.DataFrame) and X.empty):
            raise ValueError("Input data cannot be None or empty")

        # 转换为Polars DataFrame
        if isinstance(X, pd.DataFrame):
            df = pl.from_pandas(X.reset_index())
        else:
            df = X

        # 分离标识列(date/code)和数值列
        id_cols = df.select(pl.col("^.*(date|code).*$")).columns
        value_cols = df.select(pl.exclude("^.*(date|code).*$")).columns

        if not value_cols:
            raise ValueError(
                "No numeric columns found for z-score calculation")

        # 初始化结果DataFrame
        result = df.clone()

        # 为每个数值列计算zscore
        result = result.with_columns([
            self.zscore(pl.col(col), self.window).alias(col)
            for col in value_cols
        ])

        # 选择标识列和zscore结果列
        final_cols = id_cols + value_cols
        result = result.select(final_cols).fill_nan(0).fill_null(0)

        # 如果输入是pandas DataFrame，则转换回pandas
        if isinstance(X, pd.DataFrame):
            result = result.to_pandas().set_index(id_cols).sort_index()

        return result

    def zscore(self, x: pl.Expr, window: int) -> pl.Expr:
        """
        计算滚动z-score

        返回:
            标准化后的polars表达式
        """
        return (
            x - x.rolling_mean(window_size=window)
        ) / x.rolling_std(window_size=window)


# 滚动去极值
class RollingWinsorizerTransformer(TransformerMixin):
    def __init__(self, window=20):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X.reset_index())
        else:
            X = X

        # 分离标识列(date/code)和数值列
        id_cols = X.select(pl.col("^.*(date|code).*$")).columns
        value_cols = X.select(pl.exclude("^.*(date|code).*$")).columns

        result = X.with_columns(
            [
                pl.col(col).map_batches(
                    lambda x: ts_winsorize(x.to_numpy(), self.window))
                .over('code').alias(col)
                for col in value_cols
            ]).sort(id_cols)

        if isinstance(X, pd.DataFrame):
            result = result.to_pandas().set_index(id_cols).sort_index()

        return result


# 计算对数收益率
class LogReturnsTransformer(TransformerMixin):
    """
    计算对数收益率

    参数:
        pd.DataFrame 数据框

    返回: 
        有偏移的对数收益率序列（因为对下一期的预测）
    """

    def __init__(self, period: int = 1) -> None:
        super(LogReturnsTransformer, self).__init__()
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        计算对数收益率
        """
        returns = X.close.unstack().pct_change().stack()

        return returns


# 计算普通收益率
class ReturnsTransformer(TransformerMixin):
    """
    计算收益率

    参数:
        pd.DataFrame 数据框

    返回: 
        有偏移的对数收益率序列（因为对下一期的预测）
    """

    def __init__(self, period: int = 1) -> None:
        super(ReturnsTransformer, self).__init__()
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        计算收益率
        """
        returns = X.close.unstack().pct_change(self.period).shift(-1).stack()

        return returns


# 加载特征和目标变量
class Load_X_y(TransformerMixin):

    def __init__(self) -> None:
        super(Load_X_y, self).__init__()

    def fit(self, X, y):
        """
        适配方法，此处不需要进行任何操作，直接返回自身
        参数：
            X: 特征变量，类型为pandas.DataFrame
            y: 目标变量，类型为pandas.Series
        返回：
            自身实例
        """
        return self

    def transform(self, X, y):
        """
        转换方法，加载特征和目标变量
        参数：
            X: 特征变量，类型为pandas.DataFrame
            y: 目标变量，类型为pandas.Series
        返回：
            加载后的特征变量和目标变量，类型为元组
        """
        # 确保输入数据为pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X必须是pandas DataFrame类型")
            
        if not isinstance(y, (pd.DataFrame, pd.Series)):
            raise ValueError("y必须是pandas DataFrame或Series类型")

        # 获取X和y的索引
        x_index = X.index
        y_index = y.index if isinstance(y, pd.DataFrame) else y.index

        # 获取共同索引
        common_index = x_index.intersection(y_index)

        # 使用共同索引筛选数据
        X_filtered = X.loc[common_index]
        y_filtered = y.loc[common_index]

        return (X_filtered, y_filtered)
