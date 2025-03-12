class OptionSystemError(Exception):
    """期权系统基础异常类"""
    pass

class DataFetchError(OptionSystemError):
    """数据获取错误"""
    pass

class StrategyError(OptionSystemError):
    """策略执行错误"""
    pass

class PositionError(OptionSystemError):
    """持仓操作错误"""
    pass 