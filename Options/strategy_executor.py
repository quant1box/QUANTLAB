from typing import Dict
import pandas as pd
import logging
from position_manager import PositionManager
from risk_manager import RiskManager
from strategies import ParityArbitrage, BoxArbitrage, SellerStrategy

class StrategyExecutor:
    """策略执行器"""
    
    def __init__(self, position_manager: PositionManager, risk_manager: RiskManager):
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        # 初始化策略
        self.strategies = {
            'parity': ParityArbitrage(position_manager=position_manager),
            'box': BoxArbitrage(position_manager=position_manager),
            'seller': SellerStrategy(position_manager=position_manager)
        }
    
    async def execute_all_strategies(self, market_data: pd.DataFrame):
        """执行所有策略"""
        try:
            for strategy_name, strategy in self.strategies.items():
                # 检查风险限制
                if await self.risk_manager.check_strategy_limits(strategy_name):
                    # 执行策略
                    await strategy.execute(market_data)
                    logging.info(f"执行策略 {strategy_name}")
        except Exception as e:
            logging.error(f"策略执行错误: {e}") 