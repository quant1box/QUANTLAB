from typing import Dict
import asyncio
import logging
from position_manager import PositionManager
from risk_manager import RiskManager
from data_fetcher import DataFetcher
from strategy_executor import StrategyExecutor

class OptionTradingSystem:
    """期权交易系统"""
    
    def __init__(self, config: Dict):
        """初始化交易系统"""
        self.config = config
        self.position_manager = PositionManager(storage_file=config['position_file'])
        self.risk_manager = RiskManager(
            position_manager=self.position_manager,
            risk_limits=config['risk_limits']
        )
        self.data_fetcher = DataFetcher()
        self.strategy_executor = StrategyExecutor(
            position_manager=self.position_manager,
            risk_manager=self.risk_manager
        )
    
    async def run(self):
        """运行交易系统"""
        while True:
            try:
                # 获取市场数据
                market_data = await self.data_fetcher.fetch_data()
                
                # 执行策略
                await self.strategy_executor.execute_all_strategies(market_data)
                
                # 更新持仓状态
                self.position_manager.update_positions(market_data)
                
                # 检查风险
                await self.risk_manager.check_risks(market_data)
                
                await asyncio.sleep(self.config['update_interval'])
                
            except Exception as e:
                logging.error(f"系统运行错误: {e}")
                await asyncio.sleep(5) 