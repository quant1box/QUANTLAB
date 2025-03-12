import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
import talib
import duckdb

@dataclass
class CommodityScore:
    """商品分析评分结果"""
    code: str
    trend_score: float     # 趋势强度评分
    momentum_score: float  # 动量评分
    volatility_score: float  # 波动率评分
    correlation_score: float # 相关性评分
    total_score: float      # 综合评分

class CommodityAnalyzer:
    """商品分析器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化分析器
        
        参数:
        -----------
        data : pd.DataFrame
            带有多重索引(date,code)的DataFrame,包含OHLCV数据
        """
        self.data = data
        self.commodities = {
            code: group for code, group in data.groupby(level='code')
        }
        
        # 设置seaborn绘图风格
        sns.set_theme(style="whitegrid")  # 使用seaborn的网格风格
        plt.rcParams['figure.facecolor'] = 'white'  # 设置图表背景为白色
        plt.rcParams['axes.facecolor'] = 'white'    # 设置坐标区背景为白色
        plt.rcParams['grid.color'] = '#E5E5E5'      # 设置网格线颜色为浅灰色
        
        # 设置seaborn调色板
        self.colors = sns.color_palette("husl", 8)  # 使用seaborn的husl调色板
        
    def calculate_trend_metrics(self, window: int = 20) -> pd.DataFrame:
        """
        计算趋势相关指标
        
        指标说明：
        1. ADX (平均趋向指标):
           - 用于衡量趋势的强度，不考虑趋势方向
           - 值域：0-100，通常认为>25表示趋势显著
           
        2. AROON (阿隆指标):
           - 用于识别趋势的开始和结束
           - aroon_up: 衡量上升趋势的强度
           - aroon_down: 衡量下降趋势的强度
           - aroon_osc: 震荡指标，表示趋势的净强度
           
        3. DX (动向指标):
           - 衡量价格变动的方向性强度
           - 是ADX的未平滑版本
           
        4. 趋势持续性:
           - 衡量价格持续上涨或下跌的能力
           - 使用滚动窗口计算价格方向的一致性
        """
        trend_metrics = {}
        
        for code, df in self.commodities.items():
            # 添加数据检查
            # if len(df) < window:
            #     print(f"警告: {code} 数据量不足")
            #     continue
            
            # 确保数据类型正确
            df = df.astype({'high': float, 'low': float, 'close': float})
            
            # # 添加打印调试信息
            # print(f"\n处理 {code}:")
            # print("数据样本:")
            # print(df.head())
            
            adx = talib.ADX(df['high'].values, 
                           df['low'].values, 
                           df['close'].values, 
                           timeperiod=window)
            
            # print("ADX 统计:")
            # print(f"均值: {np.nanmean(adx):.2f}")
            # print(f"非空值数量: {np.sum(~np.isnan(adx))}")
            
            # 计算AROON指标
            aroon_up, aroon_down = talib.AROON(df['high'].values,
                                              df['low'].values,
                                              timeperiod=window)
            # AROON震荡指标 = 上升-下降
            aroon_osc = aroon_up - aroon_down
            
            # 计算DX (动向指标)
            dx = talib.DX(df['high'].values,
                         df['low'].values,
                         df['close'].values,
                         timeperiod=window)
            
            # 计算趋势持续性 (价格持续上涨的比例)
            close = df['close']
            trend_persistence = ((close - close.shift(window)) > 0).rolling(window).mean()
            
            # # 保存之前打印检查
            # print("\n计算结果:")
            # print({
            #     'adx_mean': np.nanmean(adx),
            #     'adx_std': np.nanstd(adx),
            #     'aroon_osc_mean': np.nanmean(aroon_osc),
            #     'dx_mean': np.nanmean(dx),
            #     'persistence': trend_persistence.mean(),
            #     'trend_quality': (np.nanmean(adx) * trend_persistence.mean())
            # })
            
            # 汇总趋势指标
            trend_metrics[code] = {
                'adx_mean': np.nanmean(adx),        # ADX平均值
                'adx_std': np.nanstd(adx),          # ADX标准差(稳定性)
                'aroon_osc_mean': np.nanmean(aroon_osc),  # AROON震荡指标平均值
                'dx_mean': np.nanmean(dx),          # DX平均值
                'persistence': trend_persistence.mean(),  # 趋势持续性
                'trend_quality': (np.nanmean(adx) * trend_persistence.mean())  # 趋势质量综合得分
            }
            
        return pd.DataFrame(trend_metrics).T
    
    def calculate_momentum_metrics(self, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算动量相关指标
        
        指标说明：
        1. 动量值：
           - 使用不同时间窗口的收益率均值
           - 反映价格变动的方向和强度
           
        2. 动量标准差：
           - 衡量动量的稳定性
           
        3. 动量夏普比：
           - 动量均值/标准差
           - 衡量单位风险下的动量收益
        """
        momentum_metrics = {}
        
        for code, df in self.commodities.items():
            returns = df['close'].pct_change()  # 计算收益率
            metrics = {}
            
            for window in windows:
                momentum = returns.rolling(window).mean()
                metrics.update({
                    f'momentum_{window}': momentum.mean(),      # 动量均值
                    f'momentum_{window}_std': momentum.std(),   # 动量标准差
                    f'momentum_{window}_sharpe': momentum.mean() / momentum.std()  # 动量夏普比
                })
            
            momentum_metrics[code] = metrics
            
        return pd.DataFrame(momentum_metrics).T
    
    def calculate_volatility_metrics(self, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算波动率相关指标
        
        指标说明：
        1. 波动率：
           - 收益率的标准差
           - 反映价格波动的剧烈程度
           
        2. 波动率的标准差：
           - 衡量波动率本身的稳定性
           
        3. 波动率稳定性：
           - 1/波动率标准差
           - 值越大表示波动率越稳定
        """
        volatility_metrics = {}
        
        for code, df in self.commodities.items():
            # # 添加数据检查
            # if len(df) < max(windows):
            #     print(f"警告: {code} 数据量不足")
            #     continue
            
            returns = df['close'].pct_change()
            
            # # 添加打印调试信息
            # print(f"\n处理 {code} 波动率:")
            # print("收益率统计:")
            # print(returns.describe())
            
            metrics = {}
            for window in windows:
                vol = returns.rolling(window).std()
                
                # # 检查计算结果
                # print(f"\n{window}日波动率统计:")
                # print(vol.describe())
                
                metrics.update({
                    f'vol_{window}': vol.mean(),
                    f'vol_{window}_std': vol.std(),
                    f'vol_stability_{window}': 1 / vol.std() if vol.std() != 0 else 0
                })
            
            volatility_metrics[code] = metrics
            
        return pd.DataFrame(volatility_metrics).T
    
    def calculate_correlation_metrics(self) -> pd.DataFrame:
        """
        计算相关性指标
        
        指标说明：
        1. 平均相关性：
           - 与其他商品的相关系数均值
           - 反映分散化效果
           
        2. 最大相关性：
           - 与其他商品的最大相关系数
           - 识别高度相关的商品对
           
        3. 相关性稳定性：
           - 1/相关系数的标准差
           - 衡量相关关系的稳定程度
        """
        # 首先确保数据是按日频重采样，避免重复值
        daily_data = self.data.groupby(['code', pd.Grouper(freq='D', level=0)])['close'].last()
        daily_data = daily_data.reset_index()
        
        # 将数据重新整理成宽表格式
        daily_matrix = daily_data.pivot(
            index='datetime',
            columns='code',
            values='close'
        )
        
        # 计算收益率
        returns = daily_matrix.pct_change()
        
        # 计算相关性矩阵
        corr_matrix = returns.corr()
        
        correlation_metrics = {}
        for code in self.commodities.keys():
            if code in corr_matrix.columns:  # 确保代码存在于相关性矩阵中
                correlations = corr_matrix[code].drop(code)
                correlation_metrics[code] = {
                    'mean_correlation': correlations.mean(),
                    'max_correlation': correlations.max(),
                    'correlation_stability': 1 / correlations.std() if correlations.std() != 0 else 0
                }
        
        return pd.DataFrame(correlation_metrics).T
    
    def score_commodities(self, 
                         weights: Dict[str, float] = None,
                         trend_threshold: float = 25,
                         vol_threshold: float = 0.02) -> List[CommodityScore]:
        """
        对商品进行综合评分
        
        参数:
        -----------
        weights : Dict[str, float]
            各维度的权重，默认使用等权重
        trend_threshold : float
            趋势强度阈值
        vol_threshold : float
            波动率阈值
            
        返回:
        --------
        List[CommodityScore] : 商品评分列表
        """
        if weights is None:
            weights = {
                'trend': 0.3,      # 趋势权重
                'momentum': 0.3,    # 动量权重
                'volatility': 0.2,  # 波动率权重
                'correlation': 0.2  # 相关性权重
            }
            
        # 计算各维度指标
        trend_metrics = self.calculate_trend_metrics().apply(lambda x:(x - x.min())/(x.max() - x.min()))
        momentum_metrics = self.calculate_momentum_metrics().apply(lambda x:(x - x.min())/(x.max() - x.min()))
        volatility_metrics = self.calculate_volatility_metrics().apply(lambda x:(x - x.min())/(x.max() - x.min()))
        correlation_metrics = self.calculate_correlation_metrics().apply(lambda x:(x - x.min())/(x.max() - x.min()))
        # print(trend_metrics,'\n',momentum_metrics,'\n',volatility_metrics,'\n',correlation_metrics)

        scores = []
        for code in self.commodities.keys():
            # 趋势度评分 (加入更多指标)
            trend_score = (
                # (trend_metrics.loc[code, 'trend_quality'] > trend_threshold) * 
                (trend_metrics.loc[code, ['adx_mean', 'aroon_osc_mean', 'dx_mean', 'persistence']].mean())
            )
            
            # 动量评分
            momentum_score = momentum_metrics.loc[code].filter(like='sharpe').mean()
            
            # 波动率评分 (低波动率得高分)
            vol_score = (
                (volatility_metrics.loc[code].filter(like='vol_stability').mean()) 
                # *
                # (volatility_metrics.loc[code].filter(like='vol_').mean() < vol_threshold)
            )
            
            # 相关性评分 (低相关性得高分)
            corr_score = (
                (1 - correlation_metrics.loc[code, 'mean_correlation']) * 
                correlation_metrics.loc[code, 'correlation_stability']
            )
            
            # 计算总分
            total_score = (
                weights['trend'] * trend_score +
                weights['momentum'] * momentum_score +
                weights['volatility'] * vol_score +
                weights['correlation'] * corr_score
            )


            scores.append(CommodityScore(
                code=code,
                trend_score=trend_score,
                momentum_score=momentum_score,
                volatility_score=vol_score,
                correlation_score=corr_score,
                total_score=total_score
            ))

                        
        return sorted(scores, key=lambda x: x.total_score, reverse=True)
    
    def select_portfolio(self, 
                        n_select: int = 5, 
                        weights: Dict[str, float] = None,
                        trend_threshold: float = 25,
                        vol_threshold: float = 0.02) -> List[str]:
        """
        Select optimal commodity portfolio
        
        Parameters:
        -----------
        n_select : int
            Number of commodities to select
        weights : Dict[str, float]
            Scoring weights
        trend_threshold : float
            Threshold for trend strength
        vol_threshold : float
            Threshold for volatility
            
        Returns:
        --------
        List[str] : List of selected commodity codes
        """
        scores = self.score_commodities(
            weights=weights,
            trend_threshold=trend_threshold,
            vol_threshold=vol_threshold
        )
        
        return [score.code for score in scores[:n_select]]
    
    def plot_metrics_comparison(self, metrics: pd.DataFrame, title: str):
        """Plot metrics comparison"""
        plt.figure(figsize=(12, 6))
        metrics.plot(kind='bar')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_trend_analysis(self, selected_codes: List[str], window: int = 20):
        """绘制趋势分析对比图"""
        n_codes = len(selected_codes)
        fig, axes = plt.subplots(3, n_codes, figsize=(6*n_codes, 15))
        
        for idx, code in enumerate(selected_codes):
            df = self.commodities[code]
            color = self.colors[idx % len(self.colors)]
            
            # 计算技术指标
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=window)
            aroon_up, aroon_down = talib.AROON(df['high'].values, df['low'].values, timeperiod=window)
            aroon_osc = aroon_up - aroon_down
            
            # 绘制价格和趋势
            axes[0, idx].plot(df.index.get_level_values(0), df['close'], label='Price', color=color)
            axes[0, idx].set_title(f'{code} Price Trend')
            axes[0, idx].legend(loc='center left', bbox_to_anchor=(-0.2, 0.5))
            axes[0, idx].grid(True, alpha=0.3)
            
            # 绘制ADX
            axes[1, idx].plot(df.index.get_level_values(0), adx, label='ADX', color=color)
            axes[1, idx].axhline(y=25, color='#C82423', linestyle='--', label='ADX Threshold', alpha=0.5)
            axes[1, idx].set_title(f'{code} ADX')
            axes[1, idx].legend(loc='center left', bbox_to_anchor=(-0.2, 0.5))
            axes[1, idx].grid(True, alpha=0.3)
            
            # 绘制Aroon Oscillator
            axes[2, idx].plot(df.index.get_level_values(0), aroon_osc, label='Aroon Osc', color=color)
            axes[2, idx].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            axes[2, idx].set_title(f'{code} Aroon Oscillator')
            axes[2, idx].legend(loc='center left', bbox_to_anchor=(-0.2, 0.5))
            axes[2, idx].grid(True, alpha=0.3)
        
        plt.suptitle('Trend Analysis Comparison', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_momentum_analysis(self, selected_codes: List[str], windows: List[int] = [5, 20, 60]):
        """
        绘制多个商品的动量分析对比图
        
        参数:
        -----------
        selected_codes : List[str]
            选中的商品代码列表
        windows : List[int]
            动量计算窗口列表
        """
        n_codes = len(selected_codes)
        fig, axes = plt.subplots(2, n_codes, figsize=(6*n_codes, 10))
        
        for idx, code in enumerate(selected_codes):
            df = self.commodities[code]
            returns = df['close'].pct_change()
            
            # 绘制价格
            axes[0, idx].plot(df.index.get_level_values(0), df['close'], label='Price')
            axes[0, idx].set_title(f'{code} Price')
            axes[0, idx].legend()
            axes[0, idx].grid(True)
            
            # 绘制不同周期的动量
            for window in windows:
                momentum = returns.rolling(window).mean()
                axes[1, idx].plot(df.index.get_level_values(0), momentum, 
                                label=f'{window}d')
            
            axes[1, idx].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1, idx].set_title(f'{code} Momentum')
            axes[1, idx].legend()
            axes[1, idx].grid(True)
        
        plt.suptitle('Momentum Analysis Comparison', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_volatility_analysis(self, selected_codes: List[str], windows: List[int] = [5, 20, 60]):
        """
        绘制多个商品的波动率分析对比图
        
        参数:
        -----------
        selected_codes : List[str]
            选中的商品代码列表
        windows : List[int]
            波动率计算窗口列表
        """
        n_codes = len(selected_codes)
        fig, axes = plt.subplots(2, n_codes, figsize=(6*n_codes, 10))
        
        for idx, code in enumerate(selected_codes):
            df = self.commodities[code]
            returns = df['close'].pct_change()
            
            # 绘制价格
            axes[0, idx].plot(df.index.get_level_values(0), df['close'], label='Price')
            axes[0, idx].set_title(f'{code} Price')
            axes[0, idx].legend()
            axes[0, idx].grid(True)
            
            # 绘制不同周期的波动率
            for window in windows:
                volatility = returns.rolling(window).std() * np.sqrt(252)  # 年化波动率
                axes[1, idx].plot(df.index.get_level_values(0), volatility, 
                                label=f'{window}d')
            
            axes[1, idx].set_title(f'{code} Volatility')
            axes[1, idx].legend()
            axes[1, idx].grid(True)
        
        plt.suptitle('Volatility Analysis Comparison', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, selected_codes: List[str]):
        """绘制相关性矩阵热力图"""
        # 获取日频数据
        daily_data = self.data.groupby(['code', pd.Grouper(freq='D', level=0)])['close'].last()
        daily_data = daily_data.reset_index()
        
        # 只选择指定的商品
        daily_data = daily_data[daily_data['code'].isin(selected_codes)]
        
        # 透视表转换
        daily_matrix = daily_data.pivot(index='datetime', columns='code', values='close')
        
        # 计算收益率和相关性
        returns = daily_matrix.pct_change()
        corr_matrix = returns.corr()
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    cmap='coolwarm',  # seaborn风格的相关性矩阵配色
                    center=0, 
                    fmt='.2f',
                    square=True,
                    cbar_kws={"shrink": .8})
        plt.title('Selected Commodities Correlation Matrix', pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_portfolio_analysis(self, selected_codes: List[str]):
        """绘制投资组合分析图"""
        # 获取选中商品的评分
        scores = self.score_commodities()
        selected_scores = [s for s in scores if s.code in selected_codes]
        
        # 转换数据为DataFrame
        score_data = pd.DataFrame([
            {
                'code': s.code,
                'Trend': s.trend_score,
                'Momentum': s.momentum_score,
                'Volatility': s.volatility_score,
                'Correlation': s.correlation_score,
                'Total': s.total_score
            }
            for s in selected_scores
        ])
        
        
        # print(score_data)
        duckdb.sql('select * from score_data').show()
        
        
        # 设置code为索引
        score_data.set_index('code', inplace=True)
        
        # 创建雷达图
        categories = ['Trend', 'Momentum', 'Volatility', 'Correlation']
        n_cats = len(categories)
        
        # 计算角度
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 确保有足够的颜色
        n_colors = len(selected_codes)
        colors = sns.color_palette("husl", n_colors)  # 根据商品数量动态生成颜色
        
        # 绘制雷达图
        ax1 = plt.subplot(121, projection='polar')
        for i, code in enumerate(selected_codes):
            values = score_data.loc[code, categories].values.flatten().tolist()
            values += values[:1]
            ax1.plot(angles, values, linewidth=2, linestyle='solid', 
                    label=code, color=colors[i])
            ax1.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax1.set_theta_offset(np.pi / 2)
        ax1.set_theta_direction(-1)
        ax1.set_rlabel_position(0)
        plt.xticks(angles[:-1], categories)
        ax1.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))
        ax1.grid(True, alpha=0.3)
        
        # 绘制总分对比条形图
        sns.barplot(
            data=score_data.reset_index(),
            y='code',
            x='Total',
            ax=ax2,
            palette=colors  # 使用相同的颜色列表
        )
        ax2.set_title('Total Score Comparison')
        
        plt.suptitle('Portfolio Analysis Dashboard', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    
