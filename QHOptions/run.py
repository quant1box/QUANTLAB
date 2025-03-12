import pandas as pd
import time
from datetime import datetime
from options_strategy import (
    OptionConfig,
    OptionTradingSystem,
    PositionInfo
)
import duckdb
from options_strategy import OptionScreener


def main():
    """运行期权交易策略"""
    
    # 1. 创建配置
    config = OptionConfig(
        underlying_types=['ETF'],
        start_date=datetime.now().strftime('%Y-%m-%d'),
        end_date=(datetime.now() + pd.Timedelta(days=90)).strftime('%Y-%m-%d'),
        min_time_value=0.1,     # 最小时间价值
        max_delta=0.6,          # Delta绝对值上限
        min_dte=7,              # 最小剩余天数
        max_dte=90,             # 最大剩余天数
        update_interval=60      # 更新间隔(秒)
    )
    
    # 2. 创建交易系统
    system = OptionTradingSystem(
        config=config,
        data_dir='./positions'  # 持仓数据保存路径
    )
    
    # # # 3. 添加示例持仓（如果有）
    # example_position = PositionInfo(
    #     code='10007321',                # 期权代码
    #     name='500ETF购12月6250',         # 期权名称
    #     entry_date='2024-11-22',        # 开仓日期
    #     entry_price=0.1252,             # 开仓价格
    #     quantity=1,                     # 持仓数量
    #     option_type='call',             # 期权类型
    #     strike=2.8,                     # 行权价
    #     expiry='2024-12-25',            # 到期日
    #     underlying='510050',            # 标的代码
    #     underlying_name='上证50ETF',     # 标的名称
    #     underlying_price=2.557,         # 标的价格
    #     initial_metrics={               # 开仓时的指标值
    #         'delta': 0.2534,
    #         'gamma': 0.8765,
    #         'theta': -0.0023,
    #         'vega': 0.0156,
    #         'impl_vol': 0.25,
    #         'time_value': 0.1252,
    #         'rate': 0.08
    #     }
    # )
    
    # system.monitor.add_position(example_position)

    
    # 4. 运行系统
    try:
        print("=" * 50)
        print(f"  Companey 龙凤飞私募基金管理有限公司  ")
        print(f"  Starting Option Trading System  ")
        print(f"  Time: {datetime.now()}  ")
        # print("=" * 50)ı
        print(f"  Monitoring {len(system.monitor.positions)} positions  ")
        print("  Press Ctrl+C to stop  ")
        print("=" * 50)
        
        system.run()
        
    except KeyboardInterrupt:
        print("\nStopping trading system...")
    except Exception as e:
        print(f"Error running trading system: {e}")
    finally:
        print("Trading system stopped")

def test_data_fetcher():
    """测试数据获取"""
    config = OptionConfig(
        underlying_types=['ETF'],
        start_date='2024-01-15',
        end_date='2024-04-15',
        min_time_value=0.1,
        max_delta=0.3
    )
    
    system = OptionTradingSystem(config)
    
    # 获取数据
    print("Fetching option data...")
    market_data = system.data_fetcher.fetch_option_data()
    
    # 打印数据信息
    print(f"\nTotal options: {len(market_data)}")
    print("\nColumns:")
    print(market_data.columns.tolist())
    
    # 按标的分组统计
    print("\nOptions by underlying:")
    print(market_data.groupby('underlying_name').size())
    
    # 按到期日分组统计
    print("\nOptions by expiry:")
    print(market_data.groupby('expiry').size())
    
    # 查看示例数据
    print("\nSample data:")
    exp_data = market_data[['name','price','theoretical_price','impl_vol','his_vol','delta','rate','dte','option_type']].head(10)
    
    if not exp_data.empty:
        duckdb.sql('select * from exp_data').show()
    else:
        print('data is empty...')
    
    return market_data


def test_strategy():
    """测试策略逻辑"""
    config = OptionConfig(
        underlying_types=['ETF'],
        start_date='2024-11-22',
        end_date='2024-12-22',
        min_time_value=0.1,
        max_delta=0.3
    )
    
    system = OptionTradingSystem(config)
    
    # 获取数据
    market_data = system.data_fetcher.fetch_option_data()
    print(OptionScreener(config=config).screen_options(options_data=market_data))

    # 执行策略
    signals = system.strategy.execute_strategy(market_data)
    
    # 打印信号
    print(f"\nGenerated {len(signals)} trading signals")
    for signal in signals:
        print("\nSignal details:")
        print(f"Code: {signal['code']}")
        print(f"Name: {signal['name']}")
        print(f"Action: {signal['action']}")
        print(f"Price: {signal['price']}")
        print(f"Score: {signal['score']:.2f}")
        print("Metrics:")
        for metric, value in signal['metrics'].items():
            print(f"  {metric}: {value}")
            
    return signals

if __name__ == "__main__":
    # 选择运行模式
    mode = input("Select mode (1: Run system, 2: Test data, 3: Test strategy): ")
    
    mode_actions = {
        '1': main,
        '2': test_data_fetcher,
        '3': test_strategy
    }
    
    action = mode_actions.get(mode)
    if action:
        result = action()  # 调用相应的函数
    else:
        print("Invalid mode") 