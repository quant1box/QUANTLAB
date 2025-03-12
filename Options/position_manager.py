import json
import pandas as pd
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class PositionInfo:
    """期权持仓信息"""
    code: str                # 期权代码
    name: str                # 期权名称
    entry_date: str          # 开仓日期
    entry_price: float       # 开仓价格
    quantity: int           # 持仓数量
    option_type: str        # 期权类型(call/put)
    strike: float           # 行权价
    expiry: str            # 到期日
    underlying: str         # 标的代码
    underlying_name: str    # 标的名称
    underlying_price: float # 标的价格
    strategy_type: str      # 策略类型
    metrics: Dict          # 风险指标

class PositionManager:
    """持仓管理器"""
    
    def __init__(self, storage_file: str):
        """初始化持仓管理器
        
        Args:
            storage_file: 持仓数据存储文件路径
        """
        self.storage_file = storage_file
        self.positions: List[PositionInfo] = []
        self.position_metrics: Dict = {}  # 存储持仓的实时指标
        self.load_positions()

    def load_positions(self):
        """从文件加载持仓信息"""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                positions_data = json.load(f)
                self.positions = [PositionInfo(**data) for data in positions_data]
            logging.info(f"加载持仓信息成功: {len(self.positions)}个持仓")
        except FileNotFoundError:
            logging.warning(f"持仓文件不存在: {self.storage_file}")
        except Exception as e:
            logging.error(f"加载持仓信息错误: {e}")

    def save_positions(self):
        """保存持仓信息到文件"""
        try:
            positions_data = [asdict(pos) for pos in self.positions]
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(positions_data, f, ensure_ascii=False, indent=2)
            logging.info("持仓信息保存成功")
        except Exception as e:
            logging.error(f"保存持仓信息错误: {e}")

    def add_position(self, position: PositionInfo):
        """添加新持仓"""
        try:
            # 检查是否已存在相同持仓
            existing = self.get_position(position.code)
            if existing:
                logging.warning(f"持仓已存在: {position.code}")
                return
                
            self.positions.append(position)
            self.save_positions()
            logging.info(f"添加持仓成功: {position.code}")
        except Exception as e:
            logging.error(f"添加持仓错误: {e}")

    def remove_position(self, code: str):
        """移除持仓"""
        try:
            self.positions = [p for p in self.positions if p.code != code]
            self.save_positions()
            logging.info(f"移除持仓成功: {code}")
        except Exception as e:
            logging.error(f"移除持仓错误: {e}")

    def get_position(self, code: str) -> Optional[PositionInfo]:
        """获取指定持仓信息"""
        return next((p for p in self.positions if p.code == code), None)

    def get_positions(self, strategy_type: str = None) -> List[PositionInfo]:
        """获取持仓列表
        
        Args:
            strategy_type: 策略类型过滤
        """
        if strategy_type:
            return [p for p in self.positions if p.strategy_type == strategy_type]
        return self.positions

    def update_positions(self, market_data: pd.DataFrame):
        """更新持仓状态
        
        Args:
            market_data: 最新市场数据
        """
        try:
            for position in self.positions:
                self._update_position_metrics(position, market_data)
            logging.info("持仓状态更新成功")
        except Exception as e:
            logging.error(f"更新持仓状态错误: {e}")

    def _update_position_metrics(self, position: PositionInfo, market_data: pd.DataFrame):
        """更新单个持仓的指标"""
        try:
            # 获取期权最新数据
            option_data = market_data[market_data['期权代码'] == position.code].iloc[0]
            
            # 计算持仓盈亏
            current_price = float(option_data['最新价'])
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_pct = round(pnl / (position.entry_price * position.quantity), 4)
            
            # 更新风险指标
            self.position_metrics[position.code] = {
                'current_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'delta': float(option_data['Delta']),
                'gamma': float(option_data['Gamma']),
                'theta': float(option_data['Theta']),
                'vega': float(option_data['Vega']),
                'impl_vol': float(option_data['隐含波动率']),
                'time_value': float(option_data['时间价值']),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logging.error(f"更新持仓指标错误 - {position.code}: {e}")

    def get_portfolio_metrics(self) -> Dict:
        """获取组合风险指标"""
        try:
            portfolio_metrics = {
                'total_pnl': 0.0,
                'total_delta': 0.0,
                'total_gamma': 0.0,
                'total_theta': 0.0,
                'total_vega': 0.0
            }
            
            # 汇总所有持仓的风险指标
            for code, metrics in self.position_metrics.items():
                position = self.get_position(code)
                if position:
                    multiplier = position.quantity
                    portfolio_metrics['total_pnl'] += metrics['pnl']
                    portfolio_metrics['total_delta'] += metrics['delta'] * multiplier
                    portfolio_metrics['total_gamma'] += metrics['gamma'] * multiplier
                    portfolio_metrics['total_theta'] += metrics['theta'] * multiplier
                    portfolio_metrics['total_vega'] += metrics['vega'] * multiplier
            
            return portfolio_metrics
            
        except Exception as e:
            logging.error(f"计算组合指标错误: {e}")
            return {}

    def get_strategy_positions(self, strategy_type: str) -> Dict:
        """获取策略相关的持仓统计
        
        Args:
            strategy_type: 策略类型
        """
        try:
            strategy_positions = [p for p in self.positions if p.strategy_type == strategy_type]
            return {
                'count': len(strategy_positions),
                'total_value': sum(p.entry_price * p.quantity for p in strategy_positions),
                'positions': strategy_positions
            }
        except Exception as e:
            logging.error(f"获取策略持仓统计错误 - {strategy_type}: {e}")
            return {}

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建持仓管理器实例
    manager = PositionManager("positions.json")
    
    # 创建测试持仓
    test_position = PositionInfo(
        code="10004398",
        name="50ETF购4月2950",
        entry_date="2024-03-15",
        entry_price=0.1252,
        quantity=1,
        option_type="call",
        strike=2.95,
        expiry="2024-04-24",
        underlying="510050",
        underlying_name="50ETF",
        underlying_price=2.851,
        strategy_type="ParityArbitrage",
        metrics={
            'delta': 0.3521,
            'gamma': 1.2456,
            'theta': -0.0023,
            'vega': 0.0156
        }
    )
    
    # 测试添加持仓
    manager.add_position(test_position)
    
    # 测试获取持仓
    positions = manager.get_positions()
    print(f"当前持仓数量: {len(positions)}") 