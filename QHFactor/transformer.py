import numpy as np
import pandas as pd
# import talib
from typing import Dict, Tuple, Union, List
from sklearn.impute import KNNImputer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from QHFactor.fn import *
import polars as pl

"""
因子处理相关函数
去极值/标准化
因子正交.etcs
"""

class ComprehensiveLabelTransformer(TransformerMixin):

    def __init__(self, period: int = 1) -> None:
        super(ComprehensiveLabelTransformer, self).__init__()
        self.period = period

    def fit(self, X):
        """
        适配方法，此处不需要进行任何操作，直接返回自身
        """
        return self

    def transform(self, X):
        """
        转换方法，计算特征并生成标签
        参数：
            X: 待转换的输入数据，类型为pandas.DataFrame
        返回：
            包含标签的DataFrame，列名为'label'
        """
        arr = [self.returns(X), self.speculation(X), self.pos_chg(X)]
        weights = [0.5, 0.3, 0.2]  # 分别给收益率、波动率和趋势度0.4/0.3/0.3的权重

        # 将特征变量按列连接，并且去除索引的第一层级，然后通过ZScoreTransformer进行标准化处理
        p = Pipeline([
            ('roll_winsorize', RollingWinsorizeTransformer(window=20)),
            ('roll_zscore', RollingZScoreTransformer(window=20))
        ])

        df = (p.transform(pd.concat(arr, axis=1).fillna(0))
              * weights).sum(axis=1).sort_index()

        # 将结果转换为DataFrame，并且将索引的第一层级向后移动self.period个位置
        return pd.DataFrame(index=df.index, data=df.unstack().shift(-self.period).stack(), columns=['lable']).dropna()

    def returns(self, data: pd.DataFrame):
        """
        计算收益率的方法
        参数：
            data: 包含时间序列数据的DataFrame
        返回：
            收益率的时间序列，类型为pandas.Series
        """
        return data.groupby(level=1, group_keys=False).apply(lambda x: np.log(x['close']).diff(self.period))

    def volatility(self, data: pd.DataFrame):
        """
        计算真实波动率的方法
        参数：
            data: 包含时间序列数据的DataFrame
        返回：
            真实波动率的时间序列，类型为pandas.Series
        """
        return data.groupby(level=1, group_keys=False).apply(
            lambda x: talib.ATR(x['high'], x['low'], x['close'], timeperiod=14)/x['close']*np.sign(x['close'] - x['open']))

    def slope(self, data: pd.DataFrame):
        """
        计算价格斜率的方法
        参数：
            data: 包含时间序列数据的DataFrame
        返回：
            价格斜率的时间序列，类型为pandas.Series
        """
        return data.groupby(level=1, group_keys=False).apply(lambda x: talib.LINEARREG_SLOPE(x['close'], timeperiod=20))

    def calculate_max_drawdown(self, prices: pd.Series):
        """
        计算最大回撤的方法
        参数：
            prices: 价格序列，类型为pandas.Series
        返回：
            最大回撤的值，类型为float
        """
        max_drawdown = 0
        peak = prices[0]

        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def calculate_rolling_max_drawdown(self, prices: pd.Series, window=20):
        """
        计算移动最大回撤的方法
        参数：
            prices: 价格序列，类型为pandas.Series
            window: 移动窗口大小，默认为20
        返回：
            移动最大回撤的时间序列，类型为pandas.DataFrame，列名为'Rolling MDD'
        """
        rolling_max_drawdown = []

        for i in range(len(prices) - window + 1):
            window_prices = prices[i: i + window]
            max_drawdown = self.calculate_max_drawdown(window_prices)
            rolling_max_drawdown.append(max_drawdown)

        result = pd.Series(rolling_max_drawdown, index=prices.index[window-1:])
        return pd.DataFrame(result, columns=['Rolling MDD'])

    def speculation(self, data: pd.DataFrame):
        """投机度"""
        return data.groupby(level=1, group_keys=False).apply(lambda x: (x['volume']/x['position'])*np.sign(x['close'] - x['open']))

    def pos_chg(self, data: pd.DataFrame):
        """持仓变化率"""
        return data.groupby(level=1, group_keys=False).apply(lambda x: x['position'].diff()*np.sign(x['close'] - x['open']))


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

        if isinstance(X, pl.DataFrame):
            df = X
        elif isinstance(X, pd.DataFrame):
            df = pl.from_pandas(X.reset_index())
        
        returns = df.with_columns(
            pl.col('close')
            .log()  # 计算对数价格
            .diff()  # 计算差分得到收益率
            .shift(-1)  # 向前移动一期，用于预测
            .over('code')  # 按code分组计算
            .alias('returns')  # 重命名列
        ).select(
            pl.exclude(['open','high','low','volume','close'])  # 只保留returns列
        ).drop_nulls()  # 删除空值
        
        return returns
        
        # p = Pipeline([
        #     ('roll_winsorize', RollingWinsorizeTransformer(window=20))
        #     # ('roll_zscore', ZScoreTransformer())
        # ])

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
        # 特征值
        features = X

        # 对数收益率
        shifted_log_returns = y

        # 找到特征变量和目标变量之间的公共索引
        common_index = pd.Index.intersection(
            features.index, shifted_log_returns.index)

        # 返回公共索引对应的特征变量和目标变量
        return features.loc[common_index,], shifted_log_returns.loc[common_index,]


class ThreeSigmaTransformer(TransformerMixin):

    def __init__(self, n=3) -> None:
        """
        初始化方法，设置标准差倍数
        参数：
            n: 标准差倍数，类型为整数，默认为3
        """
        super(ThreeSigmaTransformer, self).__init__()
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        转换方法，将超过三倍标准差范围的值进行截断
        参数：
            X: 特征变量，类型为pandas.DataFrame或numpy.ndarray
            y: 目标变量，类型为None，默认为None
        返回：
            转换后的特征变量，类型为numpy.ndarray
        """

        mean = np.mean(X)
        std = np.std(X)
        max_range = mean + self.n * std
        min_range = mean - self.n * std

        # 使用np.clip函数将超过范围的值截断为范围内的最大值或最小值
        return np.clip(X, min_range, max_range)


class ZScoreTransformer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        转换方法，将特征变量进行标准化
        参数：
            X: 特征变量，类型为pandas.DataFrame
            y: 目标变量，类型为None，默认为None
        返回：
            标准化后的特征变量，类型为pandas.DataFrame
        """
        try:
            # 对特征变量按照第一级索引进行分组，并应用标准化函数
            # 标准化函数计算了每个组的Z分数：(x - mean(x)) / std(x)
            # 并使用0填充缺失值

            return X.groupby(level=1).transform(lambda x: (x - np.mean(x)) / np.std(x)).fillna(0)
        except Exception as e:
            raise ValueError(f"Error transforming data: {e}")


class MinMaxTransformer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None,axis=0):
        """
        转换方法，将特征变量归一化到(-1,1)区间
        参数：
            X: 特征变量，类型为pandas.DataFrame
            y: 目标变量，类型为None，默认为None
        返回：
            归一化后的特征变量，类型为pandas.DataFrame
        """
        try:
            # 对特征变量按照第一级索引进行分组，并应用归一化函数
            # 归一化函数将数据映射到(-1,1)区间: 2*(x - min(x))/(max(x) - min(x)) - 1
            # 并使用0填充缺失值
            return X.groupby(level=axis).transform(lambda x: 2*(x - np.min(x))/(np.max(x) - np.min(x)) - 1).fillna(0)
        except Exception as e:
            raise ValueError(f"Error transforming data: {e}")


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
        """
        转换方法:
        1. 按level=1进行数据分组
        2. 将每个组通过apply方法传入zscore标准化函数
        3. 返回标准化后的结果数据框
        """

        return X.groupby(level=1).transform(lambda x: ts_zscore(x.to_numpy(), self.window))


class RollingWinsorizeTransformer(TransformerMixin):
    """
        实现滚动窗口Winsorize转换的转换器类

        Attributes:
            window: 滚动窗口大小
        n: Winsorize的倍数程度
    """

    def __init__(self, window=20, n=3):
        self.window = window
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """"""

        return X.groupby(level=1).transform(lambda x: ts_winsorize(x.to_numpy(), self.window))


# 斯密特正交
class SchmidtTransformer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            df = X.copy()
            df = df.dropna(axis=1)

            if df.shape[1] == 0:
                raise ValueError("所有列均为NA")

            X = df.values
            n_features = X.shape[1]

            R = np.zeros((n_features, n_features))
            Q = np.zeros(X.shape)

            for k in range(n_features):

                # 计算R[k,k]
                rkk = np.sqrt(np.dot(X[:, k], X[:, k]))

                if np.isclose(rkk, 0):
                    raise ValueError(f"第{k}列被除数为0")

                R[k, k] = rkk
                Q[:, k] = X[:, k] / R[k, k]

                for j in range(k+1, n_features):

                    # 计算R[k,j]
                    rkj = np.dot(Q[:, k], X[:, j])

                    if np.isclose(rkk, 0):
                        raise ValueError(f"第{k}列除数为0,导致R[{k},{j}]计算异常")

                    R[k, j] = rkj
                    X[:, j] -= rkj * Q[:, k]

            Q = pd.DataFrame(Q, index=df.index, columns=df.columns)

        except Exception as e:
            print("变换失败:", e)

        return Q

# 规范正交
class CanonialTransformer(TransformerMixin):
    """规范正交"""

    def __init__(self) -> None:
        super(CanonialTransformer, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        df = X.dropna(axis=1).copy()
        col_name = df.columns
        D, U = np.linalg.eig(np.dot(df.T, df))
        S = np.dot(U, np.diag(D**(-0.5)))

        Fhat = np.dot(df, S)
        Fhat = pd.DataFrame(Fhat, columns=col_name, index=X.index)
        # Fhat = pd.concat([Fhat,class_mkt],axis = 1)

        return Fhat

# 对称正交
class SymmetryTransformer(TransformerMixin):
    """对称正交"""

    def __init__(self) -> None:
        super(SymmetryTransformer, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        df = X.dropna(axis=1).copy()
        col_name = df.columns
        D, U = np.linalg.eig(np.dot(df.T, df))
        S = np.dot(U, np.diag(D**(-0.5)))

        Fhat = np.dot(df, S)
        Fhat = np.dot(Fhat, U.T)
        Fhat = pd.DataFrame(Fhat, columns=col_name, index=X.index)
        # Fhat = pd.concat([Fhat,class_mkt],axis = 1)
        return Fhat


class MultiFactorICMetrics(TransformerMixin):
    """批量计算多个因子的IC指标"""
    
    def __init__(self, period: int = 1, freq: int = 252) -> None:
        """
        参数:
            period: 预测期数，默认为1
            freq: 年化频率，默认252(交易日)
        """
        super(MultiFactorICMetrics, self).__init__()
        self.period = period
        self.freq = freq
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y):
        """
        参数:
            X: 多个因子值DataFrame，MultiIndex (date, code)，columns为因子名
            y: 目标收益率Series，MultiIndex (date, code)
            
        返回:
            Dict[str, pd.DataFrame]: 
                key: 因子名
                value: 该因子的IC指标DataFrame
        """
        results = {}
        
        # 对每个因子分别计算IC指标
        for factor in X.columns:
            factor_series = X[factor]
            metrics = self._calculate_factor_metrics(factor_series, y)
            results[factor] = metrics
            
        return results
    
    def _calculate_factor_metrics(self, factor_series: pd.Series, y: pd.Series) -> pd.DataFrame:
        """计算单个因子的所有IC指标"""
        # 确保因子值和目标变量有相同的索引
        common_index = factor_series.index.intersection(y.index)
        x = factor_series.loc[common_index]
        y = y.loc[common_index]
        
        # 按日期分组计算
        dates = x.index.get_level_values(0).unique()
        metrics = pd.DataFrame(index=dates, columns=['IC', 'RankIC', 'ICIR', 'RankICIR'])
        
        for date in dates:
            x_date = x.loc[date]
            y_date = y.loc[date]
            
            metrics.loc[date, 'IC'] = self._calculate_ic(x_date, y_date)
            metrics.loc[date, 'RankIC'] = self._calculate_rank_ic(x_date, y_date)
        
        # 计算ICIR指标
        metrics['ICIR'] = self._calculate_icir(metrics['IC'])
        metrics['RankICIR'] = self._calculate_icir(metrics['RankIC'])
        
        return metrics
    
    def _calculate_ic(self, x: pd.Series, y: pd.Series) -> float:
        return x.corr(y) if not x.empty and not y.empty else np.nan
    
    def _calculate_rank_ic(self, x: pd.Series, y: pd.Series) -> float:
        return x.rank().corr(y.rank()) if not x.empty and not y.empty else np.nan
    
    def _calculate_icir(self, ic_series: pd.Series) -> float:
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        return ic_mean / ic_std * np.sqrt(self.freq) if ic_std != 0 else 0

# 滚动IC指标
class RollingICMetrics(TransformerMixin):
    """计算滚动窗口的IC指标"""
    
    def __init__(self, window: int = 20, period: int = 1, freq: int = 252) -> None:
        """
        参数:
            window: 滚动窗口大小，默认20
            period: 预测期数，默认1
            freq: 年化频率，默认252
        """
        super(RollingICMetrics, self).__init__()
        self.window = window
        self.period = period
        self.freq = freq
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y):
        """
        参数:
            X: 因子值DataFrame，MultiIndex (date, code)
            y: 目标收益率Series，MultiIndex (date, code)
            
        返回:
            包含滚动IC指标的DataFrame
        """
        results = {}
        
        # 对每个因子分别计算滚动IC指标
        for factor in X.columns:
            factor_series = X[factor]
            metrics = self._calculate_rolling_metrics(factor_series, y)
            results[factor] = metrics
            
        return results
    
    def _calculate_rolling_metrics(self, factor_series: pd.Series, y: pd.Series) -> pd.DataFrame:
        """计算单个因子的滚动IC指标"""
        common_index = factor_series.index.intersection(y.index)
        x = factor_series.loc[common_index]
        y = y.loc[common_index]
        
        dates = x.index.get_level_values(0).unique()
        metrics = pd.DataFrame(index=dates, columns=['Rolling_IC', 'Rolling_RankIC', 'Rolling_ICIR', 'Rolling_RankICIR'])
        
        for i in range(len(dates)):
            if i < self.window:
                continue
                
            window_dates = dates[i-self.window:i]
            x_window = x.loc[window_dates]
            y_window = y.loc[window_dates]
            
            metrics.loc[dates[i], 'Rolling_IC'] = self._calculate_rolling_ic(x_window, y_window)
            metrics.loc[dates[i], 'Rolling_RankIC'] = self._calculate_rolling_rank_ic(x_window, y_window)
            
        # 计算滚动ICIR
        metrics['Rolling_ICIR'] = self._calculate_rolling_icir(metrics['Rolling_IC'])
        metrics['Rolling_RankICIR'] = self._calculate_rolling_icir(metrics['Rolling_RankIC'])
        
        return metrics
    
    def _calculate_rolling_ic(self, x: pd.Series, y: pd.Series) -> float:
        """计算窗口内的IC"""
        daily_ic = []
        for date in x.index.get_level_values(0).unique():
            x_date = x.loc[date]
            y_date = y.loc[date]
            ic = x_date.corr(y_date)
            daily_ic.append(ic)
        return np.nanmean(daily_ic)
    
    def _calculate_rolling_rank_ic(self, x: pd.Series, y: pd.Series) -> float:
        """计算窗口内的RankIC"""
        daily_rank_ic = []
        for date in x.index.get_level_values(0).unique():
            x_date = x.loc[date]
            y_date = y.loc[date]
            rank_ic = x_date.rank().corr(y_date.rank())
            daily_rank_ic.append(rank_ic)
        return np.nanmean(daily_rank_ic)
    
    def _calculate_rolling_icir(self, ic_series: pd.Series) -> float:
        """计算滚动窗口的ICIR"""
        return ic_series.rolling(window=self.window, min_periods=1).mean() / \
               ic_series.rolling(window=self.window, min_periods=1).std() * \
               np.sqrt(self.freq)

# 行业中性化
class IndustryNeutralizer(TransformerMixin):
    """行业中性化处理类"""
    
    def __init__(self):
        super(IndustryNeutralizer, self).__init__()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, factor_series: pd.Series, industry_codes: pd.Series) -> pd.Series:
        """
        对因子进行行业中性化处理

        参数:
        - factor_series: pd.Series, 因子值序列，多层索引(date, code)
        - industry_codes: pd.Series, 行业代码序列，多层索引(date, code)

        返回:
        - pd.Series: 行业中性化后的因子序列
        """
        neutralized_factors = pd.Series(index=factor_series.index, dtype=float)
        
        # 按日期进行循环
        for date in factor_series.index.get_level_values(0).unique():
            # 获取当日数据
            daily_factors = factor_series.loc[date]
            daily_industry = industry_codes.loc[date]
            
            # 创建行业哑变量矩阵
            industry_dummies = pd.get_dummies(daily_industry)
            
            # 构建回归矩阵
            X = industry_dummies
            y = daily_factors
            
            # 去除缺失值
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(y) > 0:
                # 进行线性回归
                beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                # 计算残差作为中性化后的因子值
                residuals = y - X @ beta
                neutralized_factors.loc[date] = residuals
                
        return neutralized_factors

# 市值中性化
class MarketCapNeutralizer(TransformerMixin):
    """市值中性化处理类"""
    
    def __init__(self):
        super(MarketCapNeutralizer, self).__init__()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, factor_series: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        对因子进行市值中性化处理

        参数:
        - factor_series: pd.Series, 因子值序列，多层索引(date, code)
        - market_cap: pd.Series, 市值序列，多层索引(date, code)

        返回:
        - pd.Series: 市值中性化后的因子序列
        """
        neutralized_factors = pd.Series(index=factor_series.index, dtype=float)
        
        # 对市值取对数
        log_market_cap = np.log(market_cap)
        
        # 按日期进行循环
        for date in factor_series.index.get_level_values(0).unique():
            # 获取当日数据
            daily_factors = factor_series.loc[date]
            daily_market_cap = log_market_cap.loc[date]
            
            # 构建回归矩阵
            X = pd.DataFrame({'market_cap': daily_market_cap})
            X['intercept'] = 1
            y = daily_factors
            
            # 去除缺失值
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(y) > 0:
                # 进行线性回归
                beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                # 计算残差作为中性化后的因子值
                residuals = y - X @ beta
                neutralized_factors.loc[date] = residuals
                
        return neutralized_factors

