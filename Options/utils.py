def calculate_time_value(option_price: float, underlying_price: float, 
                        strike: float, option_type: str) -> float:
    """计算时间价值"""
    intrinsic_value = max(0, underlying_price - strike) if option_type == 'call' \
                     else max(0, strike - underlying_price)
    return option_price - intrinsic_value

def calculate_moneyness(underlying_price: float, strike: float) -> float:
    """计算虚实值程度"""
    return underlying_price / strike - 1 