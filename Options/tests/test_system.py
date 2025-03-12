import unittest
import asyncio
import pandas as pd
from datetime import datetime
import logging
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import DataFetcher
from position_manager import PositionManager, PositionInfo
from risk_manager import RiskManager
from strategy_executor import StrategyExecutor
from option_system import OptionTradingSystem
from strategies.parity_arbitrage import ParityArbitrage
from strategies.box_arbitrage import BoxArbitrage
from exceptions import OptionSystemError

class TestOptionSystem(unittest.TestCase):
    """期权交易系统测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试前的准备工作"""
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 测试配置
        cls.config = {
            "position_file": "test_positions.json",
            "update_interval": 60,
            "risk_limits": {
                "max_delta": 100,
                "max_gamma": 50,
                "max_loss": -100000,
                "position_limits": {
                    "parity": 10,
                    "box": 5,
                    "seller": 20
                }
            },
            "strategy_params": {
                "parity": {
                    "min_spread": 0.01,
                    "max_positions": 10
                },
                "box": {
                    "min_spread": 0.02,
                    "max_positions": 5
                }
            }
        }
    
    def setUp(self):
        """每个测试用例前的准备"""
        self.data_fetcher = DataFetcher()
        self.position_manager = PositionManager(self.config['position_file'])
        self.risk_manager = RiskManager(
            position_manager=self.position_manager,
            risk_limits=self.config['risk_limits']
        )
        
    async def test_data_fetcher(self):
        """测试数据获取"""
        try:
            # 测试数据获取
            market_data = await self.data_fetcher.fetch_data()
            
            # 验证数据结构
            self.assertIsInstance(market_data, pd.DataFrame)
            self.assertFalse(market_data.empty)
            
            # 验证必要字段
            required_columns = ['期权代码', '期权名称', '最新价', '到期日', '行权价']
            for col in required_columns:
                self.assertIn(col, market_data.columns)
                
            logging.info("数据获取测试通过")
            
        except Exception as e:
            self.fail(f"数据获取测试失败: {e}")
            
    def test_position_manager(self):
        """测试持仓管理"""
        try:
            # 创建测试持仓
            test_position = PositionInfo(
                code="test_code",
                name="test_option",
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=1.0,
                quantity=1,
                option_type="call",
                strike=100.0,
                expiry="2024-12-31",
                underlying="test_underlying",
                underlying_name="Test ETF",
                underlying_price=100.0,
                strategy_type="ParityArbitrage",
                metrics={
                    'delta': 0.5,
                    'gamma': 0.1,
                    'theta': -0.01,
                    'vega': 0.2
                }
            )
            
            # 测试添加持仓
            self.position_manager.add_position(test_position)
            positions = self.position_manager.get_positions()
            self.assertEqual(len(positions), 1)
            
            # 测试获取持仓
            position = self.position_manager.get_position("test_code")
            self.assertIsNotNone(position)
            
            # 测试移除持仓
            self.position_manager.remove_position("test_code")
            positions = self.position_manager.get_positions()
            self.assertEqual(len(positions), 0)
            
            logging.info("持仓管理测试通过")
            
        except Exception as e:
            self.fail(f"持仓管理测试失败: {e}")
            
    async def test_risk_manager(self):
        """测试风险管理"""
        try:
            # 添加测试持仓
            test_position = PositionInfo(
                code="test_code",
                name="test_option",
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=1.0,
                quantity=1,
                option_type="call",
                strike=100.0,
                expiry="2024-12-31",
                underlying="test_underlying",
                underlying_name="Test ETF",
                underlying_price=100.0,
                strategy_type="ParityArbitrage",
                metrics={
                    'delta': 0.5,
                    'gamma': 0.1,
                    'theta': -0.01,
                    'vega': 0.2
                }
            )
            self.position_manager.add_position(test_position)
            
            # 获取市场数据
            market_data = await self.data_fetcher.fetch_data()
            
            # 测试风险检查
            await self.risk_manager.check_risks(market_data)
            
            # 测试策略限制检查
            result = await self.risk_manager.check_strategy_limits("ParityArbitrage")
            self.assertTrue(result)
            
            logging.info("风险管理测试通过")
            
        except Exception as e:
            self.fail(f"风险管理测试失败: {e}")
            
    async def test_strategy_execution(self):
        """测试策略执行"""
        try:
            # 创建策略执行器
            strategy_executor = StrategyExecutor(
                position_manager=self.position_manager,
                risk_manager=self.risk_manager
            )
            
            # 获取市场数据
            market_data = await self.data_fetcher.fetch_data()
            
            # 测试策略执行
            await strategy_executor.execute_all_strategies(market_data)
            
            logging.info("策略执行测试通过")
            
        except Exception as e:
            self.fail(f"策略执行测试失败: {e}")
            
    async def test_full_system(self):
        """测试完整系统"""
        try:
            # 创建系统实例
            system = OptionTradingSystem(self.config)
            
            # 测试系统运行
            await system.run()
            
            logging.info("系统测试通过")
            
        except Exception as e:
            self.fail(f"系统测试失败: {e}")

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptionSystem)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    # 运行测试
    asyncio.run(run_tests()) 