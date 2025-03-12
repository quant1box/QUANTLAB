import json
import time
import duckdb
import numpy as np
import pandas as pd
import akshare as ak
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple


@dataclass
class OptionConfig:
    """期权策略配置类"""
    underlying_types: List[str]  # 标的类型：ETF/股指/商品
    start_date: str
    end_date: str
    min_time_value: float = 0.1     # 最小时间价值
    max_delta: float = 0.3          # Delta限制
    min_dte: int = 7                # 最小剩余天数
    max_dte: int = 45               # 最大剩余天数
    update_interval: int = 60       # 更新间隔(秒)

@dataclass
class PositionInfo:
    """期权持仓信息"""
    code: str                   # 期权代码
    name: str                   # 期权名称
    entry_date: str             # 开仓日期
    entry_price: float          # 开仓价格
    quantity: int               # 持仓数量
    option_type: str            # 期权类型 ('call'/'put')
    strike: float               # 行权价
    expiry: str                 # 到期日
    underlying: str             # 标的代码
    underlying_name: str        # 标的名称
    underlying_price: float     # 标的价格
    initial_metrics: Dict       # 开仓时的指标值，包含：
                                        # - delta
                                        # - gamma
                                        # - theta
                                        # - vega
                                        # - impl_vol
                                        # - time_value
                                        # - rate (折溢价率)


class OptionDataFetcher:
    """期权数据获取器"""
    
    def __init__(self, config: OptionConfig):
        self.config = config
        self.cache = {}
        self.etf_codes = {
            '510050': '上证50ETF',
            '510300': '沪深300ETF',
            '510500': '中证500ETF',
            '588000': '科创50ETF',
            '588080': '科创板50ETF'
        }
        
    def fetch_option_data(self, date: str = None) -> pd.DataFrame:
        """获取期权数据
        
        Args:
            date: 日期，默认为当前交易日
            
        Returns:
            DataFrame: 期权数据，包含以下字段：
                - code: 期权代码
                - underlying: 标的代码
                - option_type: 期权类型 ('call'/'put')
                - strike: 行权价
                - expiry: 到期日
                - price: 当前价格
                - impl_vol: 隐含波动率
                - delta: Delta值
                - time_value: 时间价值
                - volume: 成交量
                - dte: 剩余天数
        """
        try:
            all_data = []
            
            # 获取ETF期权数据
            if 'ETF' in self.config.underlying_types:
                etf_data = self._fetch_etf_options()
                all_data.append(etf_data)
            
            # 获取股指期权数据
            if 'INDEX' in self.config.underlying_types:
                index_data = self._fetch_index_options()
                all_data.append(index_data)
            
            # 获取商品期权数据
            if 'COMMODITY' in self.config.underlying_types:
                commodity_data = self._fetch_commodity_options()
                all_data.append(commodity_data)
            
            # 合并所有数据
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return self._process_option_data(combined_data)
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching option data: {e}")
            return pd.DataFrame()
    
    def _fetch_etf_options(self) -> pd.DataFrame:
        """获取ETF期权数据"""
        try:
            # 1. 获取数据
            df_value = ak.option_value_analysis_em()
            df_risk = ak.option_risk_analysis_em()
            df_premium = ak.option_premium_analysis_em()
            
            # 2. 选择需要的列
            value_columns = {
                '期权代码': 'code',
                '期权名称': 'name',
                '最新价': 'price',
                '时间价值': 'time_value',
                '内在价值': 'intrinsic_value',
                '隐含波动率': 'impl_vol',
                '理论价格':'theoretical_price',
                '标的名称': 'underlying_name',
                '标的最新价':'underlying_price',
                '标的近一年波动率':'his_vol',
                '到期日': 'expiry',
            }
            
            risk_columns = {
                '期权代码': 'code',
                'Delta': 'delta',
                'Gamma': 'gamma',
                'Theta': 'theta',
                'Vega': 'vega',
            }
            
            premium_columns = {
                '期权代码': 'code',
                '行权价': 'strike',
                '折溢价率': 'rate', 
            }
            
            # 3. 预处理数据
            df_value = df_value[value_columns.keys()].rename(columns=value_columns)
            df_risk = df_risk[risk_columns.keys()].rename(columns=risk_columns)
            df_premium = df_premium[premium_columns.keys()].rename(columns=premium_columns)
            
            # 4. 合并数据
            merged_df = df_value.merge(
                df_risk[['code'] + list(set(risk_columns.values()) - {'code'})],
                on='code', 
                how='left'
            ).merge(
                df_premium[['code'] + list(set(premium_columns.values()) - {'code'})],
                on='code', 
                how='left'
            )
            
            # 5. 计算剩余天数
            merged_df['expiry'] = pd.to_datetime(merged_df['expiry'])
            current_date = pd.Timestamp.now().normalize()  # 获取当前日期（不含时间）
            merged_df['dte'] = (merged_df['expiry'] - current_date).dt.days
            
            # 6. 提取期权类型和标的代码
            merged_df['option_type'] = merged_df['name'].apply(
                lambda x: 'call' if '购' in x else 'put'
            )
            merged_df['underlying'] = merged_df['underlying_name'].map({v:k for k,v in self.etf_codes.items()})
            
            # 7. 转换数值类型
            numeric_columns = [
                'price', 'time_value', 'intrinsic_value', 'impl_vol',
                'theoretical_price', 'underlying_price', 'strike',
                'delta', 'gamma', 'theta', 'vega', 'rate', 'dte'
            ]
            
            for col in numeric_columns:
                if col in merged_df.columns:
                    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            
            # 8. 筛选有效期权（剩余天数大于0）
            merged_df = merged_df[merged_df['dte'] > 0]
            
            # # 9. 根据到期日和剩余天数筛选
            if hasattr(self.config, 'min_dte') and hasattr(self.config, 'max_dte'):
                merged_df = merged_df[
                    (merged_df['dte'] >= self.config.min_dte) &
                    (merged_df['dte'] <= self.config.max_dte)
                ]
            
            # 10. 添加时间戳
            merged_df['timestamp'] = pd.Timestamp.now()
            merged_df['expiry'] = merged_df['expiry'].apply(lambda x:x.strftime('%Y-%m-%d'))
            
            # 11. 筛选需要的标的
            if self.etf_codes:
                merged_df = merged_df[merged_df['underlying'].isin(self.etf_codes.keys())]
            
            return merged_df
            
        except Exception as e:
            print(f"Error in _fetch_etf_options: {e}")
            return pd.DataFrame()
            
    def _process_etf_option_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理ETF期权数据
        
        统一数据格式，计算必要的指标
        """
        try:
            # 重命名列
            column_mapping = {
                '期权代码': 'code',
                '买量': 'bid_volume',
                '买价': 'bid_price',
                '最新价': 'price',
                '卖价': 'ask_price',
                '卖量': 'ask_volume',
                '持仓量': 'open_interest',
                '成交量': 'volume',
                '行权价': 'strike',
                '剩余天数': 'dte',
                '隐含波动率': 'impl_vol',
                'Delta': 'delta',
                'Gamma': 'gamma',
                'Theta': 'theta',
                'Vega': 'vega',
                '理论价值': 'theoretical_price',
                '到期日': 'expiry'
            }
            
            data = data.rename(columns=column_mapping)
            
            # 提取期权类型
            data['option_type'] = data['code'].apply(
                lambda x: 'call' if x.endswith('C') else 'put'
            )
            
            # 确保数类型正确
            numeric_columns = [
                'price', 'strike', 'impl_vol', 'delta', 'gamma',
                'theta', 'vega', 'volume', 'open_interest', 'dte'
            ]
            
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 计算时间价值
            data['intrinsic_value'] = data.apply(
                lambda x: max(0, x['price'] - x['strike']) if x['option_type'] == 'call'
                else max(0, x['strike'] - x['price']),
                axis=1
            )
            data['time_value'] = data['price'] - data['intrinsic_value']
            
            # 添加当前时间戳
            data['timestamp'] = pd.Timestamp.now()
            
            return data
            
        except Exception as e:
            print(f"Error processing ETF option data: {e}")
            return pd.DataFrame()
            
    def _fetch_index_options(self) -> pd.DataFrame:
        """获取股指期权数据"""
        try:
            # 获取上证50股指期权
            df_50 = ak.option_cffex_sz50_list()
            
            # 获取沪深300股指期权
            df_300 = ak.option_cffex_hs300_list()
            
            # 合数据
            index_data = pd.concat([df_50, df_300], ignore_index=True)
            return index_data
            
        except Exception as e:
            print(f"Error fetching index options: {e}")
            return pd.DataFrame()
    
    def _fetch_commodity_options(self) -> pd.DataFrame:
        """获取商品期权数据"""
        try:
            commodity_data = []
            
            # 获取各类商品期权数据
            # 铜期权
            df_cu = ak.option_shfe_daily(symbol="cu")
            commodity_data.append(df_cu)
            
            # 黄金期权
            df_au = ak.option_shfe_daily(symbol="au")
            commodity_data.append(df_au)
            
            # 橡胶期权
            df_ru = ak.option_shfe_daily(symbol="ru")
            commodity_data.append(df_ru)
            
            # 合并所有商品期权数据
            return pd.concat(commodity_data, ignore_index=True)
            
        except Exception as e:
            print(f"Error fetching commodity options: {e}")
            return pd.DataFrame()
    
    def _process_option_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理期权数据，统一格式"""
        try:
            # 统一列名
            column_mapping = {
                '期权代码': 'code',
                '标的代码': 'underlying',
                '期权类型': 'option_type',
                '行权价': 'strike',
                '到期日': 'expiry',
                '最新价': 'price',
                '隐含波动率': 'impl_vol',
                'Delta': 'delta',
                '时间价值': 'time_value',
                '成交量': 'volume'
            }
            
            # 重命名列
            data = data.rename(columns=column_mapping)
            
            # 计算剩余天数
            data['dte'] = pd.to_datetime(data['expiry']).apply(
                lambda x: (x - pd.Timestamp.now()).days
            )
            
            # 统一期权类型表示
            data['option_type'] = data['option_type'].map(
                {'认购': 'call', '看涨': 'call', 'call': 'call',
                 '认沽': 'put', '看跌': 'put', 'put': 'put'}
            )
            
            # 确保数值类型正确
            numeric_columns = ['strike', 'price', 'impl_vol', 'delta', 
                             'time_value', 'volume', 'dte']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
            
        except Exception as e:
            print(f"Error processing option data: {e}")
            return pd.DataFrame()
    
    def get_historical_volatility(self, underlying_code: str, 
                                window: int = 252) -> float:
        """获取标的历史波动率"""
        try:
            # 获取标的历史数据
            if underlying_code.endswith('ETF'):
                hist_data = ak.fund_etf_hist_sina(symbol=underlying_code)
            else:
                hist_data = ak.stock_zh_a_hist(symbol=underlying_code)
            
            # 计算历史波动率
            returns = np.log(hist_data['收盘'].pct_change() + 1)
            hist_vol = returns.std() * np.sqrt(252)
            
            return hist_vol
            
        except Exception as e:
            print(f"Error calculating historical volatility: {e}")
            return None 
    
    def get_underlying_price(self, code: str) -> float:
        """获取标的价格"""
        try:
            # 使用 akshare 获取 ETF 实时价格
            if code in self.etf_codes:
                price_data = ak.fund_etf_spot_em()  # 东方财富ETF行情
                price = price_data[price_data['代码'] == code]['最新价'].iloc[0]
                return float(price)
            return None
        except Exception as e:
            print(f"Error getting underlying price for {code}: {e}")
            return None


class OptionScreener:
    """期权筛选器
    
    基于多个维度筛选合适的期权合约：
    1. 波动率价差 - 寻找高估期权
    2. Delta中性 - 控制方向风险
    3. 时间价值 - 确保足够时间价值
    4. 期限结构 - 控制到期时间
    5. 折溢价率 - 考虑定价偏差
    """
    
    def __init__(self, config: OptionConfig):
        self.config = config
        
    def screen_options(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """筛选满足条件的期权
        
        Args:
            options_data: 期权数据，包含必要的字段
            
        Returns:
            满足条件的期权数据
        """
        try:
            return options_data[
                # 市场价值 - 确保市场价值大于理论价值
                (options_data['price'] > options_data['theoretical_price']) &
                
                # 波动率价差 - 隐含波动率高于历史波动率
                (options_data['impl_vol']/options_data['his_vol'] > 1.05) &
                # (options_data['impl_vol'] - options_data['his_vol'] > 0.05) &
                
                # Delta中性 - 控制方向风险
                (abs(options_data['delta']) < self.config.max_delta) &
                
                # 时间价值 - 确保足够的时间价值
                (options_data['time_value'] > self.config.min_time_value) &
                
                # 期限结构 - 控制到期时间
                (options_data['dte'] >= self.config.min_dte) &
                (options_data['dte'] <= self.config.max_dte) &
                
                # 折溢价 - 期权相高估
                (options_data['rate'] > 0.05)
            ].copy()
            
        except Exception as e:
            print(f"期权筛选错误: {e}")
            return pd.DataFrame()


class PositionDataManager:
    """持仓数据管理器"""
    
    def __init__(self, data_dir: str = './positions'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.positions_file = self.data_dir / 'positions.json'
        self.metrics_file = self.data_dir / 'metrics_history.json'
        self.remove_history_file = self.data_dir / 'remove_history.json'
        
    def save_positions(self, positions: Dict[str, PositionInfo]) -> None:
        """保存持仓数据"""
        positions_data = {}
        for code, pos in positions.items():
            positions_data[code] = {
                'code': pos.code,
                'name': pos.name,
                'entry_date': pos.entry_date,
                'entry_price': pos.entry_price,
                'quantity': pos.quantity,
                'option_type': pos.option_type,
                'strike': pos.strike,
                'expiry': pos.expiry,      # 转换为字符串
                'underlying': pos.underlying,
                'underlying_name': pos.underlying_name,
                'underlying_price': pos.underlying_price,
                'initial_metrics': pos.initial_metrics
            }
        
        # 使用 ensure_ascii=False 以支持中文字符
        with open(self.positions_file, 'w', encoding='utf-8') as f:
            json.dump(positions_data, f, ensure_ascii=False, indent=4)
            
    def load_positions(self) -> Dict[str, PositionInfo]:
        """加载持仓数据"""
        if not self.positions_file.exists():
            return {}
        
        try:
            with open(self.positions_file, 'r') as f:
                positions_data = json.load(f)
                
            positions = {}
            for code, data in positions_data.items():
                # Create a complete data dictionary with all required fields
                position_data = {
                    'code': data['code'],
                    'name': data['name'],  # Ensure name is included
                    'entry_date': data['entry_date'],
                    'entry_price': data['entry_price'],
                    'quantity': data['quantity'],
                    'option_type': data['option_type'],
                    'strike': data['strike'],
                    'expiry': data['expiry'],
                    'underlying': data['underlying'],
                    'underlying_name': data['underlying_name'],  # Ensure underlying_name is included
                    'underlying_price': data['underlying_price'],  # Ensure underlying_price is included
                    'initial_metrics': data['initial_metrics'],
                }
                
                # Create PositionInfo object with all required fields
                positions[code] = PositionInfo(**position_data)
                
            return positions
            
        except Exception as e:
            print(f"Error loading positions: {e}")
            return {}
    
    def save_remove_history(self, remove_info: Dict):
        """保存移除历史记录
        
        Args:
            remove_info: 移除信息字典
        """
        try:
            # 读取现有历史记录
            if self.remove_history_file.exists():
                with open(self.remove_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
                
            # 添加新记录
            history.append(remove_info)
            
            # 保存更新后的历史记录
            with open(self.remove_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"保存移除历史记录错误: {e}")


class PositionMonitor:
    """期权持仓监控器"""
    
    def __init__(self, config: OptionConfig, data_dir: str = './positions'):
        self.config = config
        self.data_manager = PositionDataManager(data_dir)
        self.positions = self.data_manager.load_positions()
        self.history_file = Path(data_dir) / 'position_history.json'
    
    def _save_position_history(self, code: str, status: Dict):
        """保存持仓状态历史
        
        Args:
            code: 期权代码
            status: 当前状态信息
        """
        try:
            # 确保时间戳是字符串格式
            if isinstance(status['timestamp'], (pd.Timestamp, datetime)):
                status['timestamp'] = status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # 确保所有字段都是可序列化的类型
            for key in status.keys():
                if isinstance(status[key], (pd.Timestamp, datetime)):
                    status[key] = status[key].strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(status[key], (np.int64, np.float64)):
                    status[key] = int(status[key]) if isinstance(status[key], np.int64) else float(status[key])
                elif status[key] is None or (isinstance(status[key], float) and pd.isna(status[key])):
                    status[key] = 'N/A'  # 或者其他适当的默认值
            
            # 读取现有历史记录
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = {}
            
            # 初始化该期权的历史记录
            if code not in history:
                history[code] = []
            
            # 添加新的状态记录
            history[code].append(status)
            
            # 保存更新后的历史记录
            with open(self.history_file, 'w') as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"保存持仓历史记录错误 {code}: {e}")
    
    def update_positions(self, market_data: pd.DataFrame) -> Dict[str, Dict]:
        """更新持仓状态并检查预警条件"""
        alerts = {}
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for code, position in self.positions.items():
            try:
                # 检查期权是否存在于市场数据中
                position_data = market_data[market_data['code'] == code]
                if position_data.empty:
                    print(f"Warning: Position {code} not found in market data")
                    continue
                    
                current_data = position_data.iloc[0]
                
                # 检查期权是否已到期
                if current_data['dte'] <= 0:
                    self.remove_position(code, reason="到期")
                    continue
                
                # 计算持仓盈亏
                pnl = (position.entry_price - current_data['price']) * position.quantity
                pnl_pct = pnl / (position.entry_price * position.quantity)
                
                # 记录持仓状态
                position_status = {
                    'timestamp': current_time,
                    'current_price': current_data['price'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'delta': current_data['delta'],
                    'impl_vol': current_data['impl_vol'],
                    'time_value': current_data['time_value'],
                    'dte': current_data['dte']
                }
                
                # 检查预警条件
                alerts[code] = self._check_alerts(
                    position=position,
                    current_data=current_data,
                    pnl_pct=pnl_pct
                )
                
                # 保存状态历史
                self._save_position_history(code, position_status)
                
            except Exception as e:
                print(f"持仓 {code} 更新错误: {e}")
                continue
                
        return alerts
    
    def _check_alerts(self, position: PositionInfo, 
                     current_data: pd.Series,
                     pnl_pct: float) -> Dict[str, str]:
        """检查预警条件
        
        检查项目：
        1. Delta变化 - 方向风险变化
        2. 波动率变化 - 波动率变变化
        3. Theta衰减 - 时间价值衰减
        4. 止损条件 - 保护性止损
        5. 行权距离 - 距离行权价的百分比
        """
        alerts = {}
        initial = position.initial_metrics
        
        # Delta化检查
        delta_change = abs(current_data['delta'] - initial['delta'])
        if delta_change > 0.1:
            alerts['delta'] = f"Delta变化超过0.1: {delta_change:.2f}"
            
        # 波动率变化检查
        vol_change = (current_data['impl_vol'] - initial['impl_vol']) / initial['impl_vol']
        if abs(vol_change) > 0.20:
            direction = "上升" if vol_change > 0 else "下降"
            alerts['vol'] = f"隐含波动率变化超过20%: {vol_change:.1%} ({direction})"

        elif abs(vol_change) > 0.10:
            direction = "上升" if vol_change > 0 else "下降"
            alerts['vol_critical'] = f"隐含波动率变化超过10%: {vol_change:.1%} ({direction})"
        
        # 历史波动率对比检查
        vol_spread = current_data['impl_vol'] - current_data['his_vol']
        if vol_spread < 0:
            alerts['vol_spread'] = f"隐含波动率低于历史波动率: {vol_spread:.1%}"
            
        # 时间价值衰减检查
        time_value_change = (current_data['time_value'] - initial['time_value']) / initial['time_value']
        if time_value_change < -0.3:
            alerts['time_value'] = f"时间价值衰减超过30%: {-time_value_change:.1%}"

        # 新增检查：当前时间价值是否小于初始时间价值的20%
        if current_data['time_value'] < initial['time_value'] * 0.2:
            alerts['time_value_low'] = f"当前时间价值低于初始时间价值的20%: 当前时间价值 {current_data['time_value']:.4f}, 初始时间价值 {initial['time_value']:.4f}"
            
        # 新增一个检查条件：距离行权价的百分比，计算公式为 (标的价格 - 行权价格)/行权价格 × 100%
        distance_percentage = (current_data['underlying_price'] - position.strike) / position.strike * 100
        if abs(distance_percentage) <= 3:
            alerts['strike_distance'] = f"距离行权价小于3%: {distance_percentage:.2f}%"

        # 止损检查
        if pnl_pct < -0.5:
            alerts['stop_loss'] = f"触发止损条件，当前亏损: {pnl_pct:.1%}"
            
        return alerts
    
    def add_position(self, position: PositionInfo):
        """添加新持仓"""
        if not self._is_valid_position_data(position):
            print("持仓数据无效，无法添加")
            return
        
        self.positions[position.code] = position
        self.data_manager.save_positions(self.positions)
    
    def remove_position(self, code: str, reason: str = None):
        """移除持仓并更新本地文件
        
        Args:
            code: 期权代码
            reason: 移除原因(例如:'止损','到期','平仓')
        """
        try:
            if code in self.positions:
                position = self.positions[code]
                
                # 记录移除信息
                remove_info = {
                    'code': code,
                    'name': position.name,
                    'remove_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'reason': reason,
                    'entry_price': position.entry_price,
                    'exit_price': None,  # 这里可以传入实际的平仓价格
                    'holding_days': (datetime.now() - pd.to_datetime(position.entry_date)).days
                }
                
                # 从持仓字典中移除
                del self.positions[code]
                
                # 更新本地文件
                self.data_manager.save_positions(self.positions)
                
                # 记录移除历史
                self.data_manager.save_remove_history(remove_info)
                
                print(f"持仓 {code} 移除")
                print(f"移除原因: {reason}")
                print(f"持仓天数: {remove_info['holding_days']}天")
                
            else:
                print(f"未找到持仓 {code}")
                
        except Exception as e:
            print(f"移除持仓错误 {code}: {e}")

    def _is_valid_position_data(self, position: PositionInfo) -> bool:
        """验证持仓数据的有效性"""
        if not isinstance(position.code, str) or not position.code:
            print("无效的期权代码")
            return False
        if position.quantity <= 0:
            print("持仓数量必须大于0")
            return False
        if position.entry_price <= 0:
            print("开仓价格必须大于0")
            return False
        return True


class OptionStrategy:
    """期权策略类"""
    
    def __init__(self, config: OptionConfig):
        self.config = config
        self.screener = OptionScreener(config)
        
    def execute_strategy(self, market_data: pd.DataFrame) -> List[Dict]:
        """执行策略"""
        try:
            # 1. 筛选期权
            selected_options = self.screener.screen_options(market_data)
            
            if not selected_options.empty:
                duckdb.sql('select * from selected_options').show()

            # 2. 生成交易信号
            signals = self._generate_signals(selected_options)
            
            # 3. 记录信号生成时间
            for signal in signals:
                signal['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return signals
            
        except Exception as e:
            print(f"策略执行错误: {e}")
            return []

    def _generate_signals(self, options: pd.DataFrame) -> List[Dict]:
        """生成交易信号
        
        Args:
            options: 筛选后的期权数据
            
        Returns:
            List[Dict]: 交易信号列表
        """
        signals = []
        try:
            for _, option in options.iterrows():
                # 计算期权得分
                score = self._calculate_option_score(option)

                if score > 0.7:  # 得分阈值
                    # 构建交易信号
                    signal = {
                        'code': option['code'],
                        'name': option['name'],
                        'action': 'sell',  # 卖出期权
                        'quantity': self._calculate_position_size(option),
                        'price': option['price'],
                        'underlying': option['underlying'],
                        'underlying_name': option['underlying_name'],
                        'underlying_price': option['underlying_price'],
                        'strike': option['strike'],
                        'expiry': option['expiry'],
                        'dte': option['dte'],
                        'option_type': option['option_type'],
                        'metrics': {
                            'delta': option['delta'],
                            'gamma': option['gamma'],
                            'theta': option['theta'],
                            'vega': option['vega'],
                            'impl_vol': option['impl_vol'],
                            'time_value': option['time_value'],
                            'rate': option['rate'],
                            'his_vol': option['his_vol']  # 添加历史波动率
                        },
                        'score': score
                    }
                    signals.append(signal)
                    
            return signals
            
        except Exception as e:
            print(f"生成信号错误: {e}")
            return []
    
    def _calculate_option_score(self, option: pd.Series) -> float:
        """计算期权得分
        
        考虑因素：
        1. 隐含波动率水平
        2. Delta中性程度
        3. 时间价值比例
        4. 折溢价率
        5. 剩余期限适中度
        """
        try:
            # 各项指标得分
            vol_score = min(option['impl_vol'] / 0.5, 1.0)  # 隐含波动率得分
            delta_score = 1 - abs(option['delta'])  # Delta中性得分
            time_value_score = min(option['time_value'] / option['price'], 1.0)  # 时间价值得分
            rate_score = min(option['rate'] / 0.2, 1.0)  # 折溢价率得分
            
            # 剩余期限得分（越接近中间值越好）
            optimal_dte = (self.config.min_dte + self.config.max_dte) / 2
            dte_score = 1 - abs(option['dte'] - optimal_dte) / optimal_dte
            
            # 综合得分（可以调整权重）
            weights = {
                'vol': 0.3,
                'delta': 0.2,
                'time_value': 0.2,
                'rate': 0.2,
                'dte': 0.1
            }
            
            total_score = (
                weights['vol'] * vol_score +
                weights['delta'] * delta_score +
                weights['time_value'] * time_value_score +
                weights['rate'] * rate_score +
                weights['dte'] * dte_score
            )
            
            return total_score
            
        except Exception as e:
            print(f"Error calculating option score: {e}")
            return 0.0
    
    def _calculate_position_size(self, option: pd.Series) -> int:
        """算开仓数量"""
        # 这里可以实现更复杂的仓位计算逻辑
        return 1  # 默认开仓1张


class OptionTradingSystem:
    """期权交易系统
    
    能模块：
    1. 数据获取 - 实时获取期权数据
    2. 策略执行 - 生成交易信号
    3. 持仓监控 - 跟踪持仓���态
    4. 风险控制 - 管理交易风险
    """
    
    def __init__(self, config: OptionConfig, data_dir: str = './positions'):
        self.config = config
        self.strategy = OptionStrategy(config)
        self.monitor = PositionMonitor(config, data_dir)
        self.data_fetcher = OptionDataFetcher(config)
        
    def run(self):
        """运行交易系统"""
        while True:
            try:
                # 1. 获取最新市场数据
                market_data = self.data_fetcher.fetch_option_data()
                if market_data.empty:
                    print("未获取到市场数据")
                    time.sleep(self.config.update_interval)
                    continue
                
                # 2. 更新并检查持仓状态
                alerts = self.monitor.update_positions(market_data)
                if alerts:
                    self._handle_alerts(alerts)
                
                # 3. 执行策略寻找新机会
                signals = self.strategy.execute_strategy(market_data)
                
                # 4. 处理交易信号
                if signals:
                    self._process_signals(signals)
                    
                # 5. 检查持仓的止盈和止损条件
                to_remove = []  # 存储需要移除的持仓代码
                for code, position in self.monitor.positions.items():
                    position_data = market_data[market_data['code'] == code]
                    if position_data.empty:
                        print(f"Warning: Position {code} not found in market data")
                        continue  # 跳过当前循环，继续下一个持仓
                    
                    current_data = position_data.iloc[0]
                    exit_condition = self.check_exit_conditions(position, current_data['price'])
                    
                    if exit_condition == 'take_profit':
                        print(f"触发止盈条件，平仓期权: {position.code}")
                        to_remove.append((code, "止盈"))  # 记录需要移除的持仓
                    elif exit_condition == 'stop_loss':
                        print(f"触发止损条件，平仓期权: {position.code}")
                        to_remove.append((code, "止损"))  # 记录需要移除的持仓
                
                # 在迭代完成后移除持仓
                for code, reason in to_remove:
                    self.monitor.remove_position(code, reason=reason)
                
                # 6. 输出交易状态
                self._print_trading_status()
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"交易系统运行错误: {e}")
                time.sleep(self.config.update_interval)
    
    def _handle_alerts(self, alerts: Dict[str, Dict]):
        """处理预警信息"""
        for code, code_alerts in alerts.items():
            print(f"\n持仓 {code} 预警信息:")
            for alert_type, message in code_alerts.items():
                print(f"- {message}")
                
            # 处理止损
            if 'stop_loss' in code_alerts:
                self._handle_stop_loss(code)
    
    def _print_trading_status(self):
        """输出交易状态"""
        print(f"\n交易状态更新时间: {datetime.now()}")
        print(f"当前持仓数量: {len(self.monitor.positions)}")
        print("持仓详情:")
        for code, position in self.monitor.positions.items():
            print(f" - {position.name} ({position.code}): {position.quantity}张")
    
    def _handle_stop_loss(self, code: str):
        """处理止损
        
        Args:
            code: 需要止损的期权代码
        """
        try:
            position = self.monitor.positions[code]
            print(f"\n执行止损:")
            print(f"期权代码: {position.code}")
            print(f"期权名称: {position.name}")
            print(f"开仓价格: {position.entry_price}")
            print(f"持仓数量: {position.quantity}")
            
            # 从监控器中移除持仓
            self.monitor.remove_position(
                code=code,
                reason="止损"
            )
            
            print(f"止损完成")
            
        except Exception as e:
            print(f"止损处理错误 {code}: {e}")
    
    def _process_signals(self, signals: List[Dict]):
        """处理交易信号并自动添加持仓
        
        Args:
            signals: 交易信号列表
        """
        try:
            for signal in signals:
                # 检查是否已经持有该期权
                if signal['code'] in self.monitor.positions:
                    print(f"已持有期权 {signal['code']}, 跳过")
                    continue
                
                # 检查是否有足够的资金开仓
                if not self._check_margin_requirement(signal):
                    print(f"可用资金不足，跳过 {signal['code']}")
                    continue
                
                # 创建新的持仓对象
                new_position = PositionInfo(
                    code=signal['code'],
                    name=signal['name'],
                    entry_date=datetime.now().strftime('%Y-%m-%d'),
                    entry_price=signal['price'],
                    quantity=signal['quantity'],
                    option_type=signal['option_type'],
                    strike=signal['strike'],
                    expiry=signal['expiry'],
                    underlying=signal['underlying'],
                    underlying_name=signal['underlying_name'],
                    underlying_price=signal['underlying_price'],
                    initial_metrics={
                        'delta': signal['metrics']['delta'],
                        'gamma': signal['metrics']['gamma'],
                        'theta': signal['metrics']['theta'],
                        'vega': signal['metrics']['vega'],
                        'impl_vol': signal['metrics']['impl_vol'],
                        'time_value': signal['metrics']['time_value'],
                        'rate': signal['metrics']['rate'],
                        'his_vol': signal['metrics']['his_vol']
                    }
                )
                
                # 添加新持仓到监控器
                self.monitor.add_position(new_position)
                
                # 记录开仓信息
                print(f"\n新建持仓:")
                print(f"期权代码: {new_position.code}")
                print(f"期权名称: {new_position.name}")
                print(f"开仓价格: {new_position.entry_price}")
                print(f"开仓数量: {new_position.quantity}")
                print(f"期权类型: {new_position.option_type}")
                print(f"到期日期: {new_position.expiry}")
                print(f"剩余天数: {signal['dte']}")
                print(f"Delta: {new_position.initial_metrics['delta']:.4f}")
                print(f"隐含波动率: {new_position.initial_metrics['impl_vol']:.2%}")
                print(f"历史波动率: {new_position.initial_metrics['his_vol']:.2%}")
                print(f"时间价值: {new_position.initial_metrics['time_value']:.4f}")
                
        except Exception as e:
            print(f"处理交易信号错误: {e}")
    
    def _check_margin_requirement(self, signal: Dict) -> bool:
        """检查保证金要求
        
        Args:
            signal: 交易信号
            
        Returns:
            bool: 是否满足保证金要求
        """
        # 这里可以实现具体的保证金检查逻辑
        return True  # 临时返回True
    
    def check_exit_conditions(self, position: PositionInfo, current_price: float) -> str:
        """检查是否触发止盈或止损条件
        
        Args:
            position: 当前持仓信息
            current_price: 当前市场价格
            
        Returns:
            str: 触发的条件 ('take_profit', 'stop_loss', 'none')
        """
        # 设定止盈和止损的目标
        take_profit_target = position.entry_price * 0.7  # 设定20%的止盈目标（价格下跌）
        stop_loss_target = position.entry_price * 1.1     # 设定10%的止损目标（价格上涨）
        
        # 检查止盈条件
        if current_price <= take_profit_target:
            return 'take_profit'
        
        # 检查止损条件
        if current_price >= stop_loss_target:
            return 'stop_loss'
        
        # # 移动止盈逻辑
        # # 如果当前价格高于开仓价格的90%，则将止盈点上移
        # if current_price < position.entry_price * 0.9:
        #     take_profit_target = current_price * 0.9  # 将止盈点设为当前价格的90%
        
        return 'none'


