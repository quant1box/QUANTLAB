from typing import Dict
import asyncio
import logging
from option_system import OptionTradingSystem
from config import load_config
from logger import setup_logger

async def main():
    """主程序入口"""
    # 设置日志
    setup_logger('option_trading.log')
    
    # 加载配置
    config = load_config('config.json')
    
    # 初始化交易系统
    system = OptionTradingSystem(config)
    
    try:
        # 启动系统
        await system.run()
    except Exception as e:
        logging.error(f"系统运行错误: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 