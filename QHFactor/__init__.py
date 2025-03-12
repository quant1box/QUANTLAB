# # 建議更新__init__.py，更清晰地組織導出內容

# from .base.operators import *
# from .base.time_series import *
# from .factors.gtja import GTJA191
# from .factors.hft import HFTFactors
# from .calculator.batch import FactorsBatchCalculator
# from .transformers import (
#     RollingWinsorizeTransformer,
#     RollingZScoreTransformer,
#     Load_X_y
# )
# from .research.analysis import FactorAnalyzer, FactorStability
# from .optimize.selector import FactorSelector
# from .backtest.engine import BacktestEngine

# __all__ = [
#     # Base operators
#     'add', 'sub', 'mul', 'div',
    
#     # Time series
#     'ts_mean', 'ts_sum', 'ts_std',
    
#     # Factors
#     'GTJA191', 'HFTFactors',
    
#     # Calculator
#     'FactorsBatchCalculator',
    
#     # Transformers
#     'RollingWinsorizeTransformer',
#     'RollingZScoreTransformer',
#     'Load_X_y',
    
#     # Research tools
#     'FactorAnalyzer',
#     'FactorStability',
    
#     # Optimization tools
#     'FactorSelector',
    
#     # Backtest tools
#     'BacktestEngine',
# ]

# # Version info
# __version__ = '0.1.0'
