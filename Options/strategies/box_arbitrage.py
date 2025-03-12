import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .base import BaseStrategy
from position_manager import PositionManager, PositionInfo

class BoxArbitrage(BaseStrategy):
    """盒式套利策略"""
    
    def __init__(self, position_manager: PositionManager, params: Dict = None):
        super().__init__(position_manager)
        self.params = params or {
            'min_spread': 0.02,
            'max_positions': 5
        }
        
    async def execute(self, market_data: pd.DataFrame):
        """执行策略"""
        try:
            # 计算套利信号
            signals = self.calculate_signals(market_data)
            
            # 检查持仓限制
            if not self.check_position_limits():
                return
                
            # 执行交易
            for signal in signals:
                if signal['spread'] > self.params['min_spread']:
                    await self._open_box_position(signal)
                    
        except Exception as e:
            logging.error(f"盒式套利策略执行错误: {e}")
            
    def calculate_signals(self, market_data: pd.DataFrame) -> List[Dict]:
        """计算套利信号"""
        signals = []
        
        # 按到期日分组
        for _, expiry_group in market_data.groupby('到期日'):
            # 获取不同行权价的组合
            strikes = sorted(expiry_group['行权价'].unique())
            
            for i in range(len(strikes)-1):
                for j in range(i+1, len(strikes)):
                    K1, K2 = strikes[i], strikes[j]
                    
                    # 获取对应期权
                    options = expiry_group[expiry_group['行权价'].isin([K1, K2])]
                    
                    # 计算盒式套利价差
                    spread = self._calculate_box_spread(options, K1, K2)
                    
                    if spread['spread'] > self.params['min_spread']:
                        signals.append(spread)
                        
        return signals
        
    def _calculate_box_spread(self, options: pd.DataFrame, K1: float, K2: float) -> Dict:
        """计算盒式套利价差"""
        try:
            # 获取各个期权
            call1 = options[(options['行权价'] == K1) & (options['期权类型'] == '认购')].iloc[0]
            call2 = options[(options['行权价'] == K2) & (options['期权类型'] == '认购')].iloc[0]
            put1 = options[(options['行权价'] == K1) & (options['期权类型'] == '认沽')].iloc[0]
            put2 = options[(options['行权价'] == K2) & (options['期权类型'] == '认沽')].iloc[0]
            
            # 计算价差
            box_value = K2 - K1
            actual_value = (call2['最新价'] - call1['最新价']) - (put2['最新价'] - put1['最新价'])
            spread = abs(box_value - actual_value)
            
            return {
                'call1': call1,
                'call2': call2,
                'put1': put1,
                'put2': put2,
                'spread': spread,
                'box_value': box_value,
                'actual_value': actual_value
            }
            
        except Exception as e:
            logging.error(f"计算盒式套利价差错误: {e}")
            return None
            
    async def _open_box_position(self, signal: Dict):
        """开启盒式套利持仓"""
        try:
            # 创建持仓信息
            positions = []
            for option_type in ['call1', 'call2', 'put1', 'put2']:
                option = signal[option_type]
                position = PositionInfo(
                    code=option['期权代码'],
                    name=option['期权名称'],
                    entry_price=float(option['最新价']),
                    quantity=1,
                    option_type=option_type[:4],  # 'call' or 'put'
                    strategy_type='BoxArbitrage'
                )
                positions.append(position)
                
            # 添加持仓
            for position in positions:
                self.position_manager.add_position(position)
                
            logging.info(f"开启盒式套利持仓: {[p.code for p in positions]}")
            
        except Exception as e:
            logging.error(f"开启盒式套利持仓错误: {e}") 