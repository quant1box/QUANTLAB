import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime

class BacktestEngine:
    """回測引擎"""
    
    def __init__(self,
                 factors: pd.DataFrame,
                 prices: pd.DataFrame,
                 weights: Optional[pd.DataFrame] = None,
                 rebalance_freq: str = 'M',
                 commission: float = 0.0003,
                 delay: int = 1):
        """
        參數:
            factors: 因子值DataFrame
            prices: 價格數據
            weights: 權重數據(可選)
            rebalance_freq: 再平衡頻率
            commission: 交易成本
            delay: 交易延遲
        """
        self.factors = factors
        self.prices = prices
        self.weights = weights
        self.rebalance_freq = rebalance_freq
        self.commission = commission
        self.delay = delay
        
        self.positions = pd.DataFrame(0, 
                                    index=prices.index,
                                    columns=prices.columns)
        self.portfolio_value = pd.Series(1.0, index=prices.index)
        
    def run(self) -> Dict:
        """
        運行回測
        
        返回:
            回測結果字典
        """
        # 獲取再平衡日期
        rebalance_dates = self._get_rebalance_dates()
        
        # 遍歷每個再平衡日
        for date in rebalance_dates:
            # 計算目標持倉
            target_positions = self._calculate_target_positions(date)
            
            # 更新持倉
            self._update_positions(date, target_positions)
            
            # 計算收益
            self._calculate_returns(date)
            
        # 計算回測指標
        results = self._calculate_metrics()
        
        return results
    
    def _get_rebalance_dates(self) -> List[datetime]:
        """獲取再平衡日期"""
        if self.rebalance_freq == 'M':
            return pd.date_range(
                self.prices.index[0],
                self.prices.index[-1],
                freq='M'
            )
        # 可添加其他頻率...
        
    def _calculate_target_positions(self, date: datetime) -> pd.Series:
        """計算目標持倉"""
        # 如果提供了權重，直接使用
        if self.weights is not None:
            return self.weights.loc[date]
            
        # 否則基於因子值計算權重
        factor_values = self.factors.loc[date]
        
        # 簡單的等權重多空組合
        n_stocks = len(factor_values)
        weights = pd.Series(0, index=factor_values.index)
        
        # 做多排名靠前的20%，做空排名靠後的20%
        ranks = factor_values.rank()
        weights[ranks > 0.8*n_stocks] = 1/int(0.2*n_stocks)
        weights[ranks < 0.2*n_stocks] = -1/int(0.2*n_stocks)
        
        return weights
    
    def _update_positions(self, date: datetime, target_positions: pd.Series):
        """更新持倉"""
        # 考慮交易延遲
        trade_date = self.prices.index[
            self.prices.index.get_loc(date) + self.delay
        ]
        
        # 計算交易成本
        old_positions = self.positions.loc[date]
        turnover = abs(target_positions - old_positions).sum()
        cost = turnover * self.commission
        
        # 更新持倉
        self.positions.loc[trade_date:] = target_positions
        
        # 扣除交易成本
        self.portfolio_value.loc[trade_date] *= (1 - cost)
    
    def _calculate_returns(self, date: datetime):
        """計算收益"""
        # 獲取價格變化
        price_change = self.prices.pct_change()
        
        # 計算組合收益
        portfolio_return = (self.positions * price_change).sum(axis=1)
        
        # 更新組合價值
        self.portfolio_value *= (1 + portfolio_return)
    
    def _calculate_metrics(self) -> Dict:
        """計算回測指標"""
        returns = self.portfolio_value.pct_change().dropna()
        
        metrics = {}
        
        # 年化收益
        metrics['annual_return'] = returns.mean() * 252
        
        # 年化波動率
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        
        # 夏普比率
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility']
        
        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        metrics['max_drawdown'] = drawdowns.min()
        
        # 勝率
        metrics['win_rate'] = (returns > 0).mean()
        
        return metrics 