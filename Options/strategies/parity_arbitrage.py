import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .base import BaseStrategy
from position_manager import PositionManager, PositionInfo

class ParityArbitrage(BaseStrategy):
    """平价套利策略"""
    
    def __init__(self, position_manager: PositionManager, params: Dict = None):
        super().__init__(position_manager)
        self.params = params or {
            'min_spread': 0.01,
            'max_positions': 10
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
                    await self._open_arbitrage_position(signal)
                    
        except Exception as e:
            logging.error(f"平价套利策略执行错误: {e}")
            
    def calculate_signals(self, market_data: pd.DataFrame) -> List[Dict]:
        """计算套利信号"""
        signals = []
        
        # 按到期日和行权价分组
        for _, group in market_data.groupby(['到期日', '行权价']):
            call = group[group['期权类型'] == '认购'].iloc[0]
            put = group[group['期权类型'] == '认沽'].iloc[0]
            
            # 计算平价关系
            S = float(group['标的最新价'].iloc[0])
            K = float(group['行权价'].iloc[0])
            C = float(call['最新价'])
            P = float(put['最新价'])
            T = float(group['剩余天数'].iloc[0]) / 365
            r = 0.03  # 无风险利率
            
            # 计算理论价差
            theoretical = C + K * np.exp(-r * T)
            actual = P + S
            spread = abs(theoretical - actual)
            
            if spread > self.params['min_spread']:
                signals.append({
                    'call': call,
                    'put': put,
                    'spread': spread,
                    'theoretical': theoretical,
                    'actual': actual
                })
                
        return signals
        
    async def _open_arbitrage_position(self, signal: Dict):
        """开启套利持仓"""
        try:
            # 创建持仓信息
            call_position = PositionInfo(
                code=signal['call']['期权代码'],
                name=signal['call']['期权名称'],
                entry_price=float(signal['call']['最新价']),
                quantity=1,
                option_type='call',
                strategy_type='ParityArbitrage'
            )
            
            put_position = PositionInfo(
                code=signal['put']['期权代码'],
                name=signal['put']['期权名称'],
                entry_price=float(signal['put']['最新价']),
                quantity=1,
                option_type='put',
                strategy_type='ParityArbitrage'
            )
            
            # 添加持仓
            self.position_manager.add_position(call_position)
            self.position_manager.add_position(put_position)
            
            logging.info(f"开启平价套利持仓: {call_position.code}/{put_position.code}")
            
        except Exception as e:
            logging.error(f"开启套利持仓错误: {e}") 