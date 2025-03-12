from typing import Dict
import pandas as pd
import logging
from position_manager import PositionManager

class RiskManager:
    """风险管理器"""
    
    def __init__(self, position_manager: PositionManager, risk_limits: Dict):
        self.position_manager = position_manager
        self.risk_limits = risk_limits
    
    async def check_risks(self, market_data: pd.DataFrame):
        """检查风险状态"""
        try:
            # 获取当前持仓风险指标
            portfolio_metrics = self.position_manager.get_portfolio_metrics()
            
            # 检查Delta限额
            if abs(portfolio_metrics['total_delta']) > self.risk_limits['max_delta']:
                await self.handle_risk_breach('delta', portfolio_metrics['total_delta'])
            
            # 检查Gamma限额
            if abs(portfolio_metrics['total_gamma']) > self.risk_limits['max_gamma']:
                await self.handle_risk_breach('gamma', portfolio_metrics['total_gamma'])
            
            # 检查总体损失限额
            if portfolio_metrics['total_pnl'] < self.risk_limits['max_loss']:
                await self.handle_risk_breach('pnl', portfolio_metrics['total_pnl'])
                
        except Exception as e:
            logging.error(f"风险检查错误: {e}")
    
    async def handle_risk_breach(self, risk_type: str, current_value: float):
        """处理风险超限情况"""
        logging.warning(f"风险超限 - {risk_type}: {current_value}")
        # 这里可以添加预警通知、自动平仓等机制
    
    async def check_strategy_limits(self, strategy_name: str) -> bool:
        """检查策略限制"""
        # 检查策略是否可以执行
        return True 