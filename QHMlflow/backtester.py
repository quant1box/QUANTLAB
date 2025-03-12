import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict
from datetime import datetime
import empyrical as ep
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns

class MLBacktester(TransformerMixin):
    """基于机器学习预测的回测器
    
    支持多个时间周期和不同类型(分类/回归)的机器学习预测结果回测
    
    Attributes:
        freq: str, 回测时间周期 ('1d', '60min', '30min' etc.)
        n_groups: int, 分组数量
        long_short: bool, 是否允许做空
        capital: float, 初始资金
        transaction_cost: float, 交易成本
    """
    
    def __init__(self,
                 freq: str = '1d',
                 n_groups: int = 10,
                 long_short: bool = True,  # 是否允许做空
                 capital: float = 1000000,
                 transaction_cost: float = 0.0003) -> None:
        """初始化回测器
        
        Args:
            freq: 回测时间周期
            n_groups: 分组数量
            long_short: 是否允许做空
            capital: 初始资金
            transaction_cost: 交易成本
        """
        self.freq = freq
        self.n_groups = n_groups
        self.long_short = long_short
        self.capital = capital
        self.transaction_cost = transaction_cost
        
        # 存储回测结果
        self.positions = None
        self.returns = None
        self.performance = None
        self.group_returns = None
        
    def _group_by_predictions(self, 
                            predictions: pd.Series,
                            dates: pd.DatetimeIndex) -> pd.Series:
        """按日期对预测值进行分组并生成仓位
        
        Args:
            predictions: 预测值序列
            dates: 日期索引
            
        Returns:
            positions: 仓位序列
        """
        positions = pd.Series(0, index=predictions.index)
        
        # 按日期遍历
        for date in dates.unique():
            mask = dates == date
            date_preds = predictions[mask]
            
            if len(date_preds) == 0:
                continue
            
            # 按预测值分组
            try:
                groups = pd.qcut(date_preds, self.n_groups, labels=False)
                
                # 第一组做多
                positions[mask & (groups == self.n_groups - 1)] = 1
                
                # 如果允许做空，最后一组做空
                if self.long_short:
                    positions[mask & (groups == 0)] = -1
                    
            except Exception as e:
                print(f"Warning: Could not create groups for date {date}: {str(e)}")
                continue
                
        return positions
    
    def _calculate_group_returns(self,
                               predictions: pd.Series,
                               returns: pd.Series,
                               dates: pd.DatetimeIndex) -> pd.DataFrame:
        """计算各分组的收益率
        
        Args:
            predictions: 预测值序列
            returns: 实际收益率序列
            dates: 日期索引
            
        Returns:
            group_returns: 分组收益率DataFrame
        """
        group_returns = []
        
        for date in dates.unique():
            mask = dates == date
            date_preds = predictions[mask]
            date_rets = returns[mask]
            
            if len(date_preds) == 0:
                continue
            
            try:
                # 按预测值分组
                labels = pd.qcut(date_preds, self.n_groups, labels=False)
                
                # 计算每组平均收益
                group_ret = pd.DataFrame({
                    'group': labels,
                    'return': date_rets
                }).groupby('group')['return'].mean()
                
                group_ret.index = [f'G{i+1}' for i in range(self.n_groups)]
                group_returns.append(pd.DataFrame(group_ret).T)
                
            except Exception as e:
                print(f"Warning: Could not calculate group returns for date {date}: {str(e)}")
                continue
            
        return pd.concat(group_returns) if group_returns else pd.DataFrame()
    
    def _calculate_returns(self,
                         positions: pd.Series,
                         price_data: pd.DataFrame) -> pd.Series:
        """计算收益率
        
        Args:
            positions: 仓位序列
            price_data: 价格数据
            
        Returns:
            returns: 收益率序列
        """
        # 计算价格变化
        price_returns = price_data['close'].pct_change()
        
        # 计算策略收益率
        strategy_returns = positions.shift(1) * price_returns
        
        # 计算交易成本
        trades = positions.diff().abs()
        costs = trades * self.transaction_cost
        
        # 净收益率
        net_returns = strategy_returns - costs
        
        return net_returns
    
    def _calculate_performance(self, returns: pd.Series) -> Dict:
        """计算策略绩效指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            performance: 绩效指标字典
        """
        performance = {
            'Total Return': ep.cum_returns_final(returns),
            'Annual Return': ep.annual_return(returns, period=self.freq),
            'Annual Volatility': ep.annual_volatility(returns, period=self.freq),
            'Sharpe Ratio': ep.sharpe_ratio(returns, period=self.freq),
            'Max Drawdown': ep.max_drawdown(returns),
            'Calmar Ratio': ep.calmar_ratio(returns, period=self.freq),
            'Sortino Ratio': ep.sortino_ratio(returns, period=self.freq),
            'Information Ratio': ep.excess_sharpe(returns),
            'Alpha': ep.alpha(returns, returns, period=self.freq),
            'Beta': ep.beta(returns, returns, period=self.freq)
        }
        
        return performance
    
    def backtest(self,
                predictions: pd.Series,
                price_data: pd.DataFrame) -> Dict:
        """执行回测
        
        Args:
            predictions: 预测值序列
            price_data: 价格数据
            
        Returns:
            回测结果字典
        """
        # 获取日期索引
        dates = predictions.index.get_level_values(0)
        
        # 生成仓位
        self.positions = self._group_by_predictions(predictions, dates)
        
        # 计算收益率
        self.returns = self._calculate_returns(self.positions, price_data)
        
        # 计算分组收益
        price_returns = price_data['close'].pct_change()
        self.group_returns = self._calculate_group_returns(predictions, price_returns, dates)
        
        # 计算绩效
        self.performance = self._calculate_performance(self.returns)
        
        return {
            'positions': self.positions,
            'returns': self.returns,
            'group_returns': self.group_returns,
            'performance': self.performance
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制回测结果图表"""
        if self.returns is None:
            raise ValueError("No backtest results to plot. Run backtest() first.")
            
        fig, axes = plt.subplots(4, 1, figsize=(12, 20))
        
        # 累积收益率
        cum_returns = (1 + self.returns).cumprod()
        cum_returns.plot(ax=axes[0])
        axes[0].set_title('Cumulative Returns')
        axes[0].grid(True)
        
        # 仓位变化
        self.positions.plot(ax=axes[1])
        axes[1].set_title('Positions')
        axes[1].grid(True)
        
        # 回撤
        drawdown = ep.drawdown(self.returns)
        drawdown.plot(ax=axes[2])
        axes[2].set_title('Drawdown')
        axes[2].grid(True)
        
        # 分组收益率箱线图
        if self.group_returns is not None:
            sns.boxplot(data=self.group_returns.melt(), 
                       x='variable', y='value',
                       ax=axes[3])
            axes[3].set_title('Group Returns Distribution')
            axes[3].set_xlabel('Groups')
            axes[3].set_ylabel('Returns')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def get_performance_summary(self) -> pd.DataFrame:
        """获取绩效总结
        
        Returns:
            performance_df: 绩效指标DataFrame
        """
        if self.performance is None:
            raise ValueError("No performance results. Run backtest() first.")
            
        return pd.DataFrame.from_dict(self.performance, orient='index', columns=['Value']) 