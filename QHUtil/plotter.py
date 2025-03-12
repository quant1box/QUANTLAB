import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

class StrategyPlotter:
    """策略绘图工具类"""
    
    def __init__(self, returns: Union[pd.Series, pd.DataFrame], 
                 title: str = "Strategy Analysis",
                 figsize: tuple = (12, 16)):
        """
        初始化绘图工具

        Args:
            returns: 收益率序列(index必须是datetime格式)
            title: 图表标题
            figsize: 图表大小
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        
        self.returns = returns
        self.title = title
        self.figsize = figsize
        
        # 计算累计收益
        self.cum_returns = (1 + returns).cumprod()
        
        # 计算回撤
        self.drawdown = self._calculate_drawdown()
        
        # 设置绘图风格
        plt.style.use('seaborn-darkgrid')  # 使用更专业的绘图风格
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C'])

    def _calculate_drawdown(self) -> pd.Series:
        """计算回撤序列 - 改进的计算方法"""
        wealth_index = self.cum_returns
        previous_peaks = wealth_index.expanding(min_periods=1).max()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns

    def plot_strategy_summary(self, save_path: str = None):
        """绘制策略综合分析图"""
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(4, 1, figure=fig, height_ratios=[2, 1.5, 1.5, 2])
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0])
        self._plot_cumulative_returns(ax1)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1])
        self._plot_drawdown(ax2)
        
        # 3. Monthly Returns
        ax3 = fig.add_subplot(gs[2])
        self._plot_monthly_returns(ax3)
        
        # 4. Monthly Heatmap
        ax4 = fig.add_subplot(gs[3])
        self._plot_monthly_heatmap(ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def _plot_cumulative_returns(self, ax):
        """绘制累计收益曲线"""
        self.cum_returns.plot(ax=ax, color='#2878B5', linewidth=2)
        ax.set_title('Cumulative Returns', fontsize=12, pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加收益率标注
        final_return = self.cum_returns.iloc[-1] - 1
        ann_return = (1 + final_return) ** (252/len(self.returns)) - 1
        
        ax.text(0.02, 0.95, 
                f'Total Return: {final_return:.2%}\nAnnualized Return: {ann_return:.2%}', 
                transform=ax.transAxes, 
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def _plot_drawdown(self, ax):
        """改进的回撤图绘制"""
        # 使用面积图绘制回撤
        ax.fill_between(self.drawdown.index, 
                       self.drawdown.values, 
                       0, 
                       color='#C82423', 
                       alpha=0.3)
        
        # 添加回撤线
        ax.plot(self.drawdown.index, 
                self.drawdown.values, 
                color='#C82423', 
                alpha=0.5, 
                linewidth=1)
        
        ax.set_title('Drawdown', fontsize=12, pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        # 添加最大回撤标注
        max_drawdown = self.drawdown.min()
        max_drawdown_date = self.drawdown.idxmin()
        
        ax.text(0.02, 0.95, 
                f'Max Drawdown: {max_drawdown:.2%}\nDate: {max_drawdown_date:%Y-%m-%d}', 
                transform=ax.transAxes, 
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def _plot_monthly_returns(self, ax):
        """改进的月度收益柱状图"""
        monthly_returns = self.returns.resample('M').sum()
        
        colors = np.where(monthly_returns >= 0, '#2878B5', '#C82423')
        monthly_returns.plot(kind='bar', 
                           color=colors, 
                           ax=ax, 
                           alpha=0.7)
        
        ax.set_title('Monthly Returns', fontsize=12, pad=10)
        ax.set_xlabel('Month')
        ax.set_ylabel('Returns')
        
        # 优化x轴标签
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        pos_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        win_rate = pos_months / total_months
        
        ax.text(0.02, 0.95, 
                f'Win Rate: {win_rate:.2%}\nPositive Months: {pos_months}/{total_months}', 
                transform=ax.transAxes, 
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def _plot_monthly_heatmap(self, ax):
        """改进的月度收益热力图"""
        returns_matrix = self._create_returns_heatmap_data()
        
        # 使用更好的颜色方案
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        sns.heatmap(returns_matrix, 
                   cmap=cmap,
                   center=0,
                   annot=True,
                   fmt='.1%',
                   cbar_kws={'label': 'Returns'},
                   ax=ax)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=12, pad=10)
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')

    def _create_returns_heatmap_data(self) -> pd.DataFrame:
        """创建热力图数据"""
        monthly_returns = self.returns.resample('M').sum()
        returns_matrix = pd.DataFrame()
        
        # 重塑数据为年份-月份矩阵
        for year in monthly_returns.index.year.unique():
            year_data = monthly_returns[monthly_returns.index.year == year]
            returns_matrix[year] = year_data.values
            
        returns_matrix = returns_matrix.T
        returns_matrix.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(returns_matrix.columns)]
        
        return returns_matrix

    def plot_rolling_statistics(self, window: int = 60, save_path: str = None):
        """
        绘制滚动统计图表
        
        Args:
            window: 滚动窗口大小(天数)
            save_path: 图片保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        
        # 1. 滚动收益率
        rolling_returns = self.returns.rolling(window=window).mean()
        rolling_returns.plot(ax=axes[0], color='blue', label=f'{window}日滚动收益率')
        axes[0].set_title(f'{window}日滚动统计', fontsize=12)
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. 滚动波动率
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(252)
        rolling_vol.plot(ax=axes[1], color='orange', label=f'{window}日滚动波动率')
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. 滚动夏普比率
        risk_free = 0.03  # 假设无风险利率为3%
        excess_returns = self.returns - risk_free/252
        rolling_sharpe = (excess_returns.rolling(window=window).mean() * 252) / \
                        (self.returns.rolling(window=window).std() * np.sqrt(252))
        rolling_sharpe.plot(ax=axes[2], color='green', label=f'{window}日滚动夏普比率')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig 