from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional
from position_manager import PositionManager

class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
        
    @abstractmethod
    async def execute(self, market_data: pd.DataFrame):
        """执行策略"""
        pass
        
    @abstractmethod
    def calculate_signals(self, market_data: pd.DataFrame) -> Dict:
        """计算交易信号"""
        pass
        
    def check_position_limits(self) -> bool:
        """检查持仓限制"""
        pass 