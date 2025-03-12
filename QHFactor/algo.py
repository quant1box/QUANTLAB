
import pandas as pd
import talib as ta
from scipy import stats


class _Function(object):

    def __init__(self, function, name, arity, is_ts=False, params_need=None):
        self.function = function
        self.name = name
        self.arity = arity

        # 新增参数
        self.is_ts = is_ts  # bool, 代表此函数是否为时间序列函数，默认为False
        self.d = 0  # int, 时间序列回滚周期，若为时间序列函数则需要重设此参数
        self.params_need = params_need  # list, 部分TA-Lib的方法需要的固定参数及顺序

    def __call__(self, *args):
        if not self.is_ts:
            return self.function(*args)
        else:
            if self.d == 0:
                raise AttributeError("Please reset attribute 'd'")
            else:
                return self.function(*args, self.d)

    # 新增重设参数d的方法
    def set_d(self, d):
        self.d = d
        self.name += '_%d' % self.d


# delay: d天以前的x1值
def _ts_delay(x1, d):
    return pd.Series(x1).shift(d).values


ts_delay1 = _Function(function=_ts_delay, name='ts_delay', arity=1, is_ts=True)


# delta: 与 d 天以前 x1 值的差值
def _ts_delta(x1, d):
    return x1 - _ts_delay(x1, d)


ts_delta1 = _Function(function=_ts_delta, name='ts_delta', arity=1, is_ts=True)


# ts_min: 过去 d 天 x1 值构成的时序数列中最小值
def _ts_min(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).min()


ts_min1 = _Function(function=_ts_min, name='ts_min', arity=1, is_ts=True)


# 过去 d 天 x1 值构成的时序数列中最大值
def _ts_max(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).max()


ts_max1 = _Function(function=_ts_max, name='ts_max', arity=1, is_ts=True)


# 过去 d 天 x1 值构成的时序数列中最小值出现的位置
def _ts_argmin(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(lambda x: x.argmin())


ts_argmin1 = _Function(function=_ts_argmin,
                       name='ts_argmin', arity=1, is_ts=True)


# 过去 d 天 x1 值构成的时序数列中最大值出现的位置
def _ts_argmax(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(lambda x: x.argmax())


ts_argmax1 = _Function(function=_ts_argmax,
                       name='ts_argmax', arity=1, is_ts=True)


# 过去 d 天 x1 值构成的时序数列中本截面日 x1 值所处分位数
def _ts_rank(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).apply(
        lambda x: stats.percentileofscore(x, x[-1]) / 100
    )


ts_rank1 = _Function(function=_ts_rank, name='ts_rank', arity=1, is_ts=True)


def _ts_sum(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).sum()


ts_sum1 = _Function(function=_ts_sum, name='ts_sum', arity=1, is_ts=True)


def _ts_stddev(x1, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).std()


ts_stddev1 = _Function(function=_ts_stddev,
                       name='ts_stddev', arity=1, is_ts=True)


def _ts_corr(x1, x2, d):
    return pd.Series(x1).rolling(d, min_periods=int(d / 2)).corr(pd.Series(x2))


ts_corr2 = _Function(function=_ts_corr, name='ts_corr', arity=2, is_ts=True)



# 过去 d 天 x1 值构成的时序数列的变化率的平均值
def _ts_mean_return(x1, d):
    return pd.Series(x1).pct_change().rolling(d, min_periods=int(d / 2)).mean()
ts_mean_return1 = _Function(function=_ts_mean_return, name='ts_mean_return',
                            arity=1, is_ts=True)



# LINEARREG_ANGLE: 过去 d 天 x1 值序列为因变量，序列 1,…,d 为自变量的线性回归角度，属于统计学信号
ts_lr_angle1 = _Function(function=ta.LINEARREG_ANGLE, name='LR_ANGLE',
                         arity=1, is_ts=True)






# LINEARREG_INTERCEPT: 过去 d 天 x1 值序列为因变量，序列 1,…,d 为自变量的线性回归截距，属于统计学信号
ts_lr_intercept1: _Function = _Function(function=ta.LINEARREG_INTERCEPT,
                                        name='LR_INTERCEPT', arity=1, is_ts=True)





# LINEARREG_SLOPE: 过去 d 天 x1 值序列为因变量，序列 1,…,d 为自变量的线性回归斜率，属于统计学信号
ts_lr_slope1 = _Function(function=ta.LINEARREG_SLOPE, name='LR_SLOPE',
                         arity=1, is_ts=True)



"""
固定参数函数
"""

# 过去 d 天 high 序列的最大值与 low 序列的最小值的平均值
fixed_midprice = _Function(function=ta.MIDPRICE, name='midprice', arity=0, is_ts=True,params_need=['Ask', 'Bid'])


# 过去 d 天的阿隆震荡指标，属于动量信号
fixed_aroonosc = _Function(function=ta.AROONOSC, name='AROONOSC', arity=0, is_ts=True,params_need=['Ask', 'Bid'])


# 过去 d 天的威廉指标，表示的是市场属于超买还是超卖状态，属于动量信号
fixed_willr = _Function(function=ta.WILLR, name='WILLR', arity=0, is_ts=True,params_need=['Ask', 'Bid', 'AvgPrice'])


# 过去 d 天的顺势指标，测量股价是否已超出正常分布范围，属于动量信号
fixed_cci = _Function(function=ta.CCI, name='CCI', arity=0, is_ts=True,params_need=['Ask', 'Bid', 'AvgPrice'])


# 过去 d 天的平均趋向指数，指标判断盘整、震荡和单边趋势，属于动量信号
fixed_adx = _Function(function=ta.ADX, name='ADX', arity=0, is_ts=True,params_need=['Ask', 'Bid', 'AvgPrice'])


# 过去 d 天的资金流量指标，反映市场的运行趋势，属于动量信号
fixed_mfi = _Function(function=ta.MFI, name='MFI', arity=0, is_ts=True,params_need=['Ask', 'Bid', 'AvgPrice', 'volume'])


# 过去 d 天的归一化波动幅度均值，属于波动性信号
fixed_natr = _Function(function=ta.NATR, name='NATR', arity=0, is_ts=True,params_need=['Ask', 'Bid', 'AvgPrice'])


fixed_kama = _Function(function=ta.kama,arity=0,is_ts=True,params_need=['Ask','Bid','AvgPrice'])

"""
定义函数集
"""
_ts_function_map = {
    'ts_delay': ts_delay1,
    'ts_delta': ts_delta1,
    'ts_min': ts_min1,
    'ts_max': ts_max1,
    'ts_argmin': ts_argmin1,
    'ts_argmax': ts_argmax1,
    'ts_rank': ts_rank1,
    'ts_stddev': ts_stddev1,
    'ts_corr': ts_corr2,
    'ts_mean_return': ts_mean_return1,

    'DEMA': ts_dema1,
    'KAMA': ts_kama1,
    'MA': ts_ma1,
    'MIDPOINT': ts_midpoint1,
    'BETA': ts_beta2,
    'LR_ANGLE': ts_lr_angle1,
    'LR_INTERCEPT': ts_lr_intercept1,
    'LR_SLOPE': ts_lr_slope1,
    'HT': ts_ht1
}


"""
固定参数函数集
"""
_fixed_function_map = {
    'MIDPRICE': fixed_midprice,
    'AROONOSC': fixed_aroonosc,
    'WILLR': fixed_willr,
    'CCI': fixed_cci,
    'ADX': fixed_adx,
    'MFI': fixed_mfi,
    'NATR': fixed_natr
}