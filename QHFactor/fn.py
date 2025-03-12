
import numpy as np
from numba import jit, njit


@njit
def returns(open: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    计算当日收益率
    """
    return close/open - 1


@njit
def adv(volume: np.ndarray, n: int = 20) -> np.ndarray:
    """
    计算过去n个交易日的平均成交量
    """

    adv_values = np.empty_like(volume)
    adv_values[:] = np.nan

    for i in range(n, len(volume)):
        adv_values[i] = np.mean(volume[i - n+1: i+1])

    return adv_values


@njit
def vwap(close: np.ndarray, volume: np.ndarray, n: int = 20) -> np.ndarray:
    """
    计算移动加权平均价
    """
    vwap = np.empty_like(close)
    vwap[:] = np.nan

    for i in range(n, len(close)):
        close_array = close[i-n+1: i+1]
        volume_array = volume[i - n+1: i+1]

        vwap[i] = np.sum(close_array*volume_array) / \
            np.sum(volume_array) if np.sum(volume_array) != 0 else 0

    return vwap


# -------------------------------------------------
# 基础函数功能
# --------------------------------------------------
@njit
def add(x, y) -> np.ndarray:
    """"""
    return np.add(x, y)


@njit
def sub(x, y) -> np.ndarray:
    """"""
    return np.subtract(x, y)


@njit
def mul(x, y) -> np.ndarray:
    """点乘"""
    return np.multiply(x, y)


@njit
def div(x, y) -> np.ndarray:
    """点除"""
    return np.divide(x, y)


@njit
def abs(x) -> np.ndarray:
    """绝对值"""
    return np.abs(x)


@njit
def sqrt(x) -> np.ndarray:
    """开方"""
    return np.sqrt(x)


@njit
def log(x) -> np.ndarray:
    """对数"""
    return np.log(x)


@njit
def inv(x) -> np.ndarray:
    """相反数"""
    return np.invert(x, dtype=float)


@njit
def reciprocal(x) -> np.ndarray:
    """倒数"""
    return np.reciprocal(x, dtype=float)


@njit
def maximum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最大值"""

    return np.maximum(x, y)


@njit
def minimum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最小值"""

    return np.minimum(x, y)


@njit
def sign(x: np.ndarray) -> np.ndarray:
    """取符号"""
    return np.sign(x)


@njit
def sequence(n: int = 10) -> np.ndarray:
    """"""
    return np.arange(1, n + 1)


@njit
def scale(x: np.ndarray, k=1) -> np.ndarray:
    """"""
    return x*k/np.abs(x).sum()


# ----------------------------------------------------
# 時序算子
# -----------------------------------------------------
@njit
def ts_sum(x: np.ndarray, n: int = 20) -> np.ndarray:
    """rolling sum"""

    tssum = np.empty_like(x)
    tssum[:] = np.nan

    for i in range(n, len(x)):
        tssum[i] = np.sum(x[i - n+1:i+1])

    return tssum


@njit
def ts_sumac(x: np.ndarray, n: int = 20) -> np.ndarray:
    """计算前n项的滚动累加和"""

    tssumac = np.empty_like(x)
    current_sum = 0

    # 初始化前n项
    for i in range(n):
        current_sum += x[i]
        tssumac[i] = current_sum

    # 从第n项开始进行滚动累加和
    for i in range(n, len(x)):
        current_sum += x[i]
        current_sum -= x[i - n]
        tssumac[i] = current_sum

    return tssumac


@njit
def ts_rank(x: np.ndarray, n: int = 10) -> np.ndarray:
    """rolling rank"""
    rank = np.empty_like(x, dtype=np.float64)
    rank[:] = np.nan  # 初始化为 NaN

    for i in range(n - 1, len(x)):
        window = x[i - n + 1:i + 1]
        if np.all(np.isnan(window)):
            rank[i] = np.nan
        else:
            rank[i] = np.argsort(np.argsort(window))[-1] + 1  # 计算排名

    return rank


@njit
def ts_mean(x: np.ndarray, n: int = 20) -> np.ndarray:
    """rolling mean"""

    tsmean = np.full_like(x, fill_value=np.nan, dtype=np.float64)

    for i in range(n - 1, len(x)):
        tsmean[i] = np.mean(x[i - n+1: i+1])

    return tsmean


@njit
def ts_sma(x: np.ndarray, n: int, m: int):
    """"""
    Y = np.zeros_like(x)

    if np.isnan(x[0]):
        Y[0] = np.nan
    else:
        Y[0] = x[0]  # 初始值设置为第一个数据点

    for i in range(1, len(x)):
        if i < n:  # 当前周期小于N时，使用普通平均值计算
            Y[i] = np.nanmean(x[:i+1])
        else:
            if np.isnan(x[i]):
                Y[i] = Y[i-1]  # 如果当前值是nan，保持前一个值
            else:
                if np.isnan(Y[i-1]):
                    Y[i] = x[i]
                else:
                    Y[i] = (m * x[i] + (n - m) * Y[i-1]) / n

    return Y


def ts_wma(x: np.array, n: int) -> np.ndarray:
    """
    计算A前n期样本加权平均值权重 0.9i,(i 表示样本距离当前时点的间隔)
    """
    wma = np.full_like(x, fill_value=np.nan, dtype=np.float32)

    w = 0.9*np.arange(n)
    for i in range(n-1, len(x)):
        wma[i] = np.sum(x[i - n + 1:i + 1] * w) / np.sum(w)

    return wma


@njit
def ts_stddev(x: np.ndarray, n: int = 20) -> np.ndarray:
    """rolling stddev"""

    tsstd = np.zeros_like(x)

    for i in range(n, len(x)):
        tsstd[i] = np.std(x[i - n+1:i+1])

    return tsstd


@njit
def ts_corr(x: np.ndarray, y: np.ndarray, n: int = 20) -> np.ndarray:
    """Calculate rolling correlation"""
    correlation = np.empty(len(x), dtype=np.float64)
    correlation[:] = np.nan  # 初始化为 NaN

    for i in range(n - 1, len(x)):
        x_window = x[i - n + 1:i + 1]
        y_window = y[i - n + 1:i + 1]

        # 计算相关性，处理 NaN 值
        if np.count_nonzero(~np.isnan(x_window)) < 2 or np.count_nonzero(~np.isnan(y_window)) < 2:
            correlation[i] = np.nan
        else:
            correlation[i] = np.corrcoef(x_window, y_window)[0, 1]

    return correlation


@njit
def ts_var(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Calculate rolling variance."""

    variance = np.empty(len(x), dtype=np.float64)

    for i in range(n, len(x)):
        variance[i] = np.var(x[i - n + 1:i + 1])

    return np.nan_to_num(variance, 0.)


@njit
def ts_cov(x: np.ndarray, y: np.ndarray, n: int = 20) -> np.ndarray:
    """Calculate rolling  covariance"""

    covariance = np.full_like(x, fill_value=np.nan, dtype=np.float64)

    for i in range(n-1, len(x)):
        x_n = x[i-n+1: i+1]
        y_n = y[i-n+1: i+1]
        covariance[i] = np.cov(x_n, y_n, ddof=0)[0, 1]

    return covariance


@njit
def ts_skew(x: np.ndarray, n: int = 20):

    skew_value = np.empty_like(x)

    for i in range(n, len(x)):

        data = x[i - n+1:i+1]
        k_mean = np.mean(data)
        k_var = np.var(data)

        skew_value[i] = np.mean((data - k_mean)**3)

    return np.round(skew_value, 6)


@njit
def ts_kurt(x: np.ndarray, n: int = 20):

    kurt_value = np.empty_like(x)

    for i in range(n, len(x)):

        data = x[i - n+1:i+1]
        s_mean = np.mean(data)
        s_var = np.var(data)

        kurt_value[i] = np.mean((data - s_mean)**4)/np.power(s_var, 2)

    return np.round(kurt_value, 6)


@njit
def delta(x: np.ndarray, n: int = 1) -> np.ndarray:
    """ 
    """
    da = np.full_like(x, fill_value=np.nan, dtype=np.float32)

    if x.shape[0] > n:
        da[n:] = x[n:] - x[:-n]

    return da


@njit
def delay(x: np.ndarray, n: int = 1) -> np.array:
    """"""
    dy = np.full_like(x,fill_value=np.nan,dtype=np.float32)

    if n > 0:
        dy[n:] = x[:-n]
    elif n < 0:
        dy[:n] = x[-n:]

    return dy


@njit
def rank(x: np.ndarray):
    """
    """
    # 获取数据长度
    n = len(x)

    # 使用 argsort 方法获取排名
    sorted_indices = x.argsort()

    # 计算排名
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(n)

    # 计算百分位
    pct_rank = ranks / (n - 1)

    return pct_rank


@njit
def ts_max(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Calculate rolling maximum"""
    max_value = np.empty_like(x, dtype=np.float32)
    max_value[:] = np.nan  # 初始化为 NaN

    for i in range(n - 1, len(x)):
        window = x[i - n + 1:i + 1]
        if np.count_nonzero(~np.isnan(window)) > 0:
            max_value[i] = np.nanmax(window)  # 使用 nanmax 处理 NaN 值

    return max_value


@njit
def ts_min(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Calculate rolling minimum"""
    min_value = np.empty_like(x, dtype=np.float32)
    min_value[:] = np.nan  # 初始化为 NaN

    for i in range(n - 1, len(x)):
        window = x[i - n + 1:i + 1]
        if np.count_nonzero(~np.isnan(window)) > 0:
            min_value[i] = np.nanmin(window)  # 使用 nanmin 处理 NaN 值

    return min_value


@njit
def ts_argmax(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Calculate rolling  argmax"""

    argmax = np.zeros_like(x)
    for i in range(n, len(x)):
        argmax[i] = np.argmax(x[i - n + 1:i + 1]) + 1

    return argmax


@njit
def ts_argmin(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Calculate rolling  argmin"""

    argmin = np.zeros_like(x)
    for i in range(n, len(x)):
        argmin[i] = np.argmin(x[i - n + 1:i + 1]) + 1

    return argmin


@njit
def ts_highday(x: np.ndarray, n: int = 20) -> np.ndarray:
    """计算前 n 期时间序列中最大值距离当前时点的间隔"""

    highday = np.zeros_like(x)
    for i in range(n, len(x)):

        x_n = x[i - n + 1:i + 1]
        highday[i] = len(x_n) - np.argmax(x_n) + 1

    return highday


@njit
def ts_lowday(x: np.ndarray, n: int = 20) -> np.ndarray:
    """计算前 n 期时间序列中最小值距离当前时点的间隔"""

    lowday = np.zeros_like(x)

    for i in range(n, len(x)):

        x_n = x[i - n + 1:i + 1]
        lowday[i] = len(x_n) - np.argmin(x_n) + 1

    return lowday


@njit
def ts_prod(x: np.ndarray, n: int = 10) -> np.ndarray:
    """"""
    prod = np.zeros_like(x)

    for i in range(n, len(x)):
        prod[i] = np.prod(x[i - n+1:i+1])

    return prod


@njit
def ts_zscore(x: np.ndarray, n: int = 20):
    """"""
    tszscore = np.empty_like(x, dtype=np.float32)
    tszscore[:] = np.nan

    for i in range(n, len(x)):
        x_n = x[i - n:i]
        x_mean = np.mean(x_n)
        x_std = np.std(x_n)

        tszscore[i] = ((x[i] - x_mean)/x_std) if x_std != 0 else 0

    return tszscore


@njit
def ts_winsorize(x: np.ndarray, n: int = 20, n_stds: int = 3):
    """"""

    tsws = np.empty_like(x)
    for i in range(0, len(x)):

        if i < n:
            tsws[i] = x[i]
        else:
            x_n = x[i - n:i]
            median = np.median(x_n)
            mad = np.median(np.abs(x_n - median))

            lower_bound = median - n_stds * mad
            upper_bound = median + n_stds * mad

            tsws[i] = np.where((x[i] >= upper_bound), upper_bound,
                               np.where((x[i] <= lower_bound), lower_bound, x[i]))

    return tsws


@njit
def count(condition, n: int = 10) -> np.ndarray:
    """"""
    counter = np.zeros(len(condition))
    for i in range(n, len(condition)):
        counter[i] = np.sum(condition[i - n+1:i+1])

    return counter


@njit
def decay_linear(x: np.ndarray, n: int = 10) -> np.ndarray:
    """
    对 X 序列计算移动平均加权，其中权重对应 n,n-1,…,1 (权重和为 1)
    """

    N = len(x)
    weights = np.arange(1, n+1, 1)  # 生成权重数组，从 d 到 1
    weights_sum = np.sum(weights)
    weighted_sum = np.zeros(N)

    for i in range(n - 1, N):
        weighted_sum[i] = np.sum(x[i - n + 1:i + 1] * weights) / weights_sum

    return weighted_sum


@njit
def reg_beta(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """回歸係數"""

    beta = np.zeros_like(x, dtype=np.float64)

    for i in range(n, len(x)):

        x_n = x[i - n+1:i+1]
        y_n = y[i - n+1:i+1]
        beta[i] = np.polyfit(x_n, y_n, 1)[0]

    return beta


@njit
def reg_resi(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """回歸殘差"""
    resi = np.zeros_like(x)

    for i in range(n, len(x)):
        x_n = x[i - n + 1:i + 1]
        y_n = y[i - n + 1:i + 1]

        k, v = np.polyfit(x_n, y_n, 1)
        y_pred = k * x_n + v

        # 計算殘差
        residuals = y_n - y_pred
        resi[i] = residuals[-1]  # 保存整个窗口内的最后一个残差值

    return resi


@njit
def custom_linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = np.sum((x - x_mean) * (y - y_mean))
    variance = np.sum((x - x_mean) ** 2)
    beta = covariance / variance
    return beta


@njit
def reg_beta(x: np.ndarray, y: np.ndarray, n: int = 10) -> np.ndarray:

    beta = np.full_like(x, fill_value=np.nan, dtype=np.float64)

    for i in range(n - 1, len(x)):
        x_n = x[i - n + 1:i + 1]
        y_n = y[i - n + 1:i + 1]
        beta[i] = custom_linear_regression(x_n, y_n)
    return beta
