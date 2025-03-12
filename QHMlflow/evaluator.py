
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score)

import pandas as pd
from typing import Union, Dict
import empyrical as ep


def evaluate_model_classification(model, X_test, y_test):
    """
    评估分类模型

    参数：
      model: 分类模型
      X_test: 测试集
      y_test: 测试集标签

    返回：
      模型的分数以及其他分类任务的评估指标
    """
    # 预测
    y_pred = model.predict(X_test)

    # 准确度
    accuracy = accuracy_score(y_test, y_pred)

    # 精确度
    precision = precision_score(y_test, y_pred, average='weighted')

    # 召回率
    recall = recall_score(y_test, y_pred, average='weighted')

    # F1 分数
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 混淆矩阵
    confusion_mat = confusion_matrix(y_test, y_pred)

    # 分类报告
    class_report = classification_report(y_test, y_pred)

    # ROC-AUC（仅适用于二分类问题）
    if len(set(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        roc_auc = None

    # 打印或返回评估指标
    print("准确度:", accuracy)
    print("精确度:", precision)
    print("召回率:", recall)
    print("F1 分数:", f1)
    print("混淆矩阵:\n", confusion_mat)
    print("分类报告:\n", class_report)

    if roc_auc is not None:
        print("ROC-AUC:", roc_auc)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_mat,
        'classification_report': class_report,
        'roc_auc': roc_auc
    }


def evaluate_model_regression(model, X_test, y_test):
    """
    评估回归模型

    参数：
      model: 回归模型
      X_test: 测试集
      y_test: 测试集目标值

    返回：
      模型的分数以及其他回归任务的评估指标
    """
    # 预测
    y_pred = model.predict(X_test)

    # 均方误差
    mse = mean_squared_error(y_test, y_pred)

    # 均方根误差
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # 平均绝对误差
    mae = mean_absolute_error(y_test, y_pred)

    # R^2 分数
    r2 = r2_score(y_test, y_pred)

    # 打印或返回评估指标
    print("均方误差 (MSE):", mse)
    print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)
    print("R^2 分数:", r2)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def performance_indicators(returns: Union[pd.Series, pd.DataFrame], period: str = 'daily', to_df: bool = False) -> Union[Dict, pd.Series]:
    # start_date = pd.Timestamp(min(returns.index)).strftime('%Y%m%d')
    # end_date = pd.Timestamp(max(returns.index)).strftime('%Y%m%d')
    start_date = int(min(returns.index).strftime('%Y%m%d'))
    end_date = int(max(returns.index).strftime('%Y%m%d'))

    annual_return = ep.annual_return(returns, period=period)
    cumulative_returns = ep.cum_returns_final(returns=returns)
    annual_volatility = ep.annual_volatility(returns, period=period)
    sharpe_ratio = ep.sharpe_ratio(returns, period=period)
    calmar_ratio = ep.calmar_ratio(returns, period=period)
    stability = ep.stability_of_timeseries(returns=returns)
    max_drawdown = ep.max_drawdown(returns)
    omega_ratio = ep.omega_ratio(returns)
    sortino_ratio = ep.sortino_ratio(returns, period=period)
    tail_ratio = ep.tail_ratio(returns)
    daily_var = ep.value_at_risk(returns)

    dt = {
        'Start date': start_date,
        'End date': end_date,
        f'Annual Return': round(annual_return, 4),
        f'Cumulative Returns': round(cumulative_returns, 4),
        f'Annual Volatility': round(annual_volatility, 4),
        f'Sharpe Ratio': round(sharpe_ratio, 4),
        f'Calmar Ratio': round(calmar_ratio, 4),
        f'Stability': round(stability, 4),
        f'Max Drawdown': round(max_drawdown, 4),
        f'Omega Ratio': round(omega_ratio, 4),
        f'Sortino Ratio': round(sortino_ratio, 4),
        f'Tail Ratio': round(tail_ratio, 4),
        f'Daily Value at Risk': round(daily_var, 4)
    }

    if to_df:
        return pd.DataFrame(index=dt.keys(), data=dt.values())[0]

    return dt
