import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from sklearn.base import TransformerMixin

class FactorAnalyzer(TransformerMixin):
    """因子分析器"""
    
    def __init__(self, 
                 periods: Union[int, List[int]] = [1, 5, 10, 20],
                 quantiles: int = 5,
                 freq: str = 'D'):
        """
        參數:
            periods: 預測期數
            quantiles: 分位數數量
            freq: 頻率 'D'日頻 'W'周頻 'M'月頻
        """
        self.periods = periods if isinstance(periods, list) else [periods]
        self.quantiles = quantiles
        self.freq = freq
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        分析因子
        
        參數:
            X: 因子值 DataFrame
            y: 收益率序列
            
        返回:
            包含各項分析結果的字典
        """
        results = {}
        
        # IC分析
        results['ic'] = self._analyze_ic(X, y)
        
        # 分層收益分析
        results['returns'] = self._analyze_layered_returns(X, y)
        
        # 換手率分析
        results['turnover'] = self._analyze_turnover(X)
        
        return results
        
    def _analyze_ic(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """IC分析"""
        ic_results = pd.DataFrame()
        
        for period in self.periods:
            # 計算IC
            ic = self._calculate_ic(X, y, period)
            ic_results[f'IC_{period}D'] = ic
            
            # 計算RankIC
            rank_ic = self._calculate_rank_ic(X, y, period)
            ic_results[f'RankIC_{period}D'] = rank_ic
            
            # 計算ICIR
            ic_results[f'ICIR_{period}D'] = self._calculate_icir(ic)
            ic_results[f'RankICIR_{period}D'] = self._calculate_icir(rank_ic)
        
        return ic_results
    
    def _analyze_layered_returns(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """分層收益分析"""
        returns_results = {}
        
        for period in self.periods:
            # 計算分位數收益
            quantile_returns = self._calculate_quantile_returns(X, y, period)
            returns_results[f'returns_{period}D'] = quantile_returns
            
            # 計算多空組合收益
            ls_returns = self._calculate_long_short_returns(quantile_returns)
            returns_results[f'ls_returns_{period}D'] = ls_returns
        
        return pd.DataFrame(returns_results)
    
    def _analyze_turnover(self, X: pd.DataFrame) -> pd.DataFrame:
        """換手率分析"""
        turnover = pd.DataFrame()
        
        # 計算各分位數換手率
        for q in range(self.quantiles):
            turnover[f'q{q+1}'] = self._calculate_quantile_turnover(X, q)
            
        return turnover
    
    def _calculate_ic(self, X: pd.DataFrame, y: pd.Series, period: int) -> pd.Series:
        """計算IC"""
        shifted_y = y.shift(-period)
        return X.corrwith(shifted_y)
    
    def _calculate_rank_ic(self, X: pd.DataFrame, y: pd.Series, period: int) -> pd.Series:
        """計算RankIC"""
        shifted_y = y.shift(-period)
        return X.rank().corrwith(shifted_y.rank())
    
    def _calculate_icir(self, ic: pd.Series) -> float:
        """計算ICIR"""
        return ic.mean() / ic.std() if ic.std() != 0 else 0
    
    def _calculate_quantile_returns(self, X: pd.DataFrame, y: pd.Series, period: int) -> pd.DataFrame:
        """計算分位數收益"""
        quantiles = pd.qcut(X, self.quantiles, labels=False)
        shifted_y = y.shift(-period)
        
        return pd.DataFrame({
            'quantile': quantiles,
            'returns': shifted_y
        }).groupby('quantile')['returns'].mean()
    
    def _calculate_long_short_returns(self, quantile_returns: pd.Series) -> float:
        """計算多空組合收益"""
        return quantile_returns.iloc[-1] - quantile_returns.iloc[0]
    
    def _calculate_quantile_turnover(self, X: pd.DataFrame, quantile: int) -> pd.Series:
        """計算分位數換手率"""
        quantile_positions = (pd.qcut(X, self.quantiles, labels=False) == quantile)
        return (quantile_positions != quantile_positions.shift(1)).mean()

class FactorStability(TransformerMixin):
    """因子穩定性分析"""
    
    def __init__(self, 
                 windows: List[int] = [20, 60, 120],
                 freq: str = 'D'):
        """
        參數:
            windows: 滾動窗口大小列表
            freq: 頻率
        """
        self.windows = windows
        self.freq = freq
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> Dict:
        """
        分析因子穩定性
        
        參數:
            X: 因子值DataFrame
            
        返回:
            包含穩定性分析結果的字典
        """
        results = {}
        
        # 自相關性分析
        results['autocorr'] = self._analyze_autocorr(X)
        
        # 因子值穩定性
        results['value_stability'] = self._analyze_value_stability(X)
        
        # 排序穩定性
        results['rank_stability'] = self._analyze_rank_stability(X)
        
        return results
        
    def _analyze_autocorr(self, X: pd.DataFrame) -> pd.DataFrame:
        """分析因子自相關性"""
        autocorr = pd.DataFrame()
        
        for window in self.windows:
            autocorr[f'autocorr_{window}D'] = X.autocorr(lag=window)
            
        return autocorr
    
    def _analyze_value_stability(self, X: pd.DataFrame) -> pd.DataFrame:
        """分析因子值穩定性"""
        stability = pd.DataFrame()
        
        for window in self.windows:
            # 計算滾動標準差
            rolling_std = X.rolling(window=window).std()
            stability[f'value_std_{window}D'] = rolling_std
            
            # 計算變異係數
            rolling_mean = X.rolling(window=window).mean()
            stability[f'cv_{window}D'] = rolling_std / rolling_mean
            
        return stability
    
    def _analyze_rank_stability(self, X: pd.DataFrame) -> pd.DataFrame:
        """分析因子排序穩定性"""
        rank_stability = pd.DataFrame()
        
        for window in self.windows:
            # 計算排序相關係數
            rank_corr = X.rank().rolling(window=window).corr(
                X.rank().shift(1))
            rank_stability[f'rank_corr_{window}D'] = rank_corr
            
        return rank_stability 