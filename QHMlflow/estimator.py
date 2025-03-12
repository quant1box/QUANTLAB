import json
import optuna
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, accuracy_score, log_loss, r2_score)

from QHData.config import Config

# from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor


# 计算换手率
def calculate_turnover_rate(y_pred: pd.DataFrame, top_n: int = 100) -> float:
    """计算换手率

    Args:
        y_pred: 预测值 DataFrame，索引为 MultiIndex(date, stock)
        top_n: 每天选择的前N名股票

    Returns:
        float: 换手率均值
    """
    if not isinstance(y_pred, pd.DataFrame):
        raise TypeError("y_pred must be a pandas DataFrame")

    if not isinstance(y_pred.index, pd.MultiIndex):
        raise ValueError("y_pred must have a MultiIndex (date, stock)")

    try:
        y_pred = y_pred.sort_index(level=0)

        # 获取每日前N支股票
        top_n_daily = {
            date: group.nlargest(min(top_n, len(group)), 0)
            .index.get_level_values(1).tolist()
            for date, group in y_pred.groupby(level=0)
        }

        dates = sorted(top_n_daily.keys())
        if len(dates) < 2:
            return 0.0

        # 计算每日换手率
        turnover_rates = {
            dates[i]: len(set(top_n_daily[dates[i-1]]) -
                          set(top_n_daily[dates[i]])) / top_n
            for i in range(1, len(dates))
        }

        return np.mean(list(turnover_rates.values()))

    except Exception as e:
        raise ValueError(f"Error calculating turnover rate: {e}")


# ------------------------------------------------------

class OptimizerConfig:
    """公共的超参数优化配置类"""

    def __init__(self, study_name: str, direction: str = "minimize"):
        """
        初始化优化配置类

        Args:
            study_name (str): Optuna研究的名称
            direction (str): 优化方向，'minimize' 或 'maximize'
        """
        self.study_name = study_name
        self.direction = direction

    def create_study(self):
        """创建Optuna研究"""

        # storage_name = f"sqlite:///{self.study_name}.db"  # 数据库存储路径
        storage_name = Config.get_postgresql_uri()

        return optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),  # 使用TPE采样器
            study_name=self.study_name,
            direction=self.direction,
            storage=storage_name,
            pruner=optuna.pruners.HyperbandPruner(),  # 使用Hyperband剪枝器
            load_if_exists=True  # 如果研究已存在，则加载
        )

    # 计算均方误差
    def mse(self, y_true, y_pred):
        """均方误差损失函数

        Args:
            y_true (array-like): 真实值
            y_pred (array-like): 预测值

        Returns:
            float: 计算得到的均方误差
        """
        return mean_squared_error(y_true, y_pred)

    def binary_cross_entropy(self, y_true, y_pred):
        """二元交叉熵损失函数

        Args:
            y_true (array-like): 真实标签
            y_pred (array-like): 预测概率

        Returns:
            float: 计算得到的二元交叉熵
        """

        return log_loss(y_true, y_pred)

    # 计算一致性相关系数 CCC
    def ccc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算一致性相关系数 (Concordance Correlation Coefficient, CCC)

        CCC衡量两组数据之间的一致性程度，取值范围为[-1, 1]：
        - 1表示完全一致
        - 0表示无关
        - -1表示完全不一致

        Parameters
        ----------
        y_true : np.ndarray
            真实值数组
        y_pred : np.ndarray
            预测值数组

        Returns
        -------
        float
            一致性相关系数值

        Raises
        ------
        ValueError
            当输入数组长度不一致时抛出
        TypeError
            当输入不是numpy数组时抛出
        """
        # 输入验证
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")

        if y_true.shape != y_pred.shape:
            raise ValueError("Arrays must have the same shape")

        # 确保输入是一维数组
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # 计算均值和标准差
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)

        # 计算协方差
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

        # 计算CCC
        numerator = 2 * covariance
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        # 处理分母为0的情况
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def objective_function(self, trial, X, y):
        """定义损失函数

        Args:
            trial (optuna.Trial): 当前的Optuna试验
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量

        Raises:
            NotImplementedError: 子类应实现此方法
        """
        raise NotImplementedError("Subclasses should implement this method.")

    # 获取最佳参数
    def get_best_params(self, study):
        """获取最佳超参数

        Args:
            study (optuna.Study): Optuna研究对象

        Returns:
            dict: 最佳超参数字典
        """
        return study.best_params

    # 获取最佳损失函数值
    def get_best_value(self, study):
        """获取最佳目标值

        Args:
            study (optuna.Study): Optuna研究对象

        Returns:
            float: 最佳目标值
        """
        return study.best_value


class LGBMRollingRegressor(TransformerMixin):
    """LGBM 回归器"""

    def __init__(self) -> None:
        super(LGBMRollingRegressor, self).__init__()

    def objective(self, trial, X, y):
        # 参数网格优化
        params = {
            'objective': 'regression',
            'importance_type': 'gain',
            # 使用gbdt而不是goss，因为gbdt在金融时序数据上通常更稳定可靠
            "boosting_type": 'gbdt',

            # 叶子节点数，对于金融数据，适当减小以避免过拟合
            "num_leaves": trial.suggest_int("num_leaves", 15, 50, step=5),

            # 树的数量，金融数据通常需要足够的树来捕捉复杂模式
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),

            # 学习率，金融数据通常需要较小的学习率以获得更稳定的结果
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.03, log=True),

            # 树的深度，金融数据易过拟合，控制深度很重要
            "max_depth": trial.suggest_int("max_depth", 3, 7, 1),

            # 最小子样本数，增大以提高模型对噪声的鲁棒性
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 120, step=10),

            # 行采样和列采样，对于金融数据的高噪声特性很有帮助
            "subsample": trial.suggest_float("subsample", 0.5, 0.9, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9, step=0.05),
            
            # 添加特征分箱参数，处理金融数据中的异常值和离群点
            "max_bin": trial.suggest_int("max_bin", 200, 500, step=50),
            
            # 正则化参数，金融数据通常需要更强的正则化
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1e-1),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 1e-1),
            
            # 添加处理类别特征的参数
            "cat_smooth": trial.suggest_int("cat_smooth", 10, 100),
            
            # 添加path smoothing参数，减少方差
            "path_smooth": trial.suggest_int("path_smooth", 0, 10),
            
            # 添加特征分数阈值，过滤低重要性特征
            "feature_fraction_seed": 42,
            
            # 添加early stopping参数
            "early_stopping_rounds": 50,
            
            # 添加处理缺失值的参数
            "use_missing": True,
            "zero_as_missing": False,
            
            "verbosity": -1,
            "random_state": 42
        }

        # 使用更多的交叉验证折数
        tscv = TimeSeriesSplit(n_splits=5)  # test_size确保每个测试集大小一致
        cv_scores = []

        for idx, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # LGBM建模
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test)

            # 添加early_stopping和verbose_eval参数
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                # early_stopping_rounds=50,
                # verbose_eval=False
            )

            # 模型预测
            y_pred = model.predict(X_test)

            # 同时考虑MSE和方向准确率
            mse_score = mean_squared_error(y_test, y_pred)
            direction_accuracy = np.mean((y_test > 0) == (y_pred > 0))

            # 综合评分（可以调整权重）
            combined_score = 0.7 * (-mse_score) + 0.3 * direction_accuracy

            # # 根据预测值的置信度调整权重
            # confidence = model.predict_proba(X_test)
            # weight_mse = np.clip(confidence, 0.6, 0.8)
            # weight_dir = 1 - weight_mse
            # combined_score = weight_mse * (-mse_score) + weight_dir * direction_accuracy

            cv_scores.append(combined_score)

        return np.mean(cv_scores)

    def optimizer(self, study_name: str = "lgb_rolling_reg", X=None, y=None):

        # Create the Optuna study
        # storage_name = "sqlite:///{}.db".format(study_name)
        storage_name = Config.get_postgresql_uri()

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name,
            direction="minimize",
            storage=storage_name,
            pruner=optuna.pruners.HyperbandPruner(),
            load_if_exists=True
        )

        # Optimize the hyperparameters
        study.optimize(lambda trial: self.objective(
            trial, X, y), n_trials=1000)

        # Get the best hyperparameters
        best_params = study.best_params
        print(f'best paras: {study.best_params}')
        print(f'best value: {study.best_value}')

        # Train the LightGBM model with the best hyperparameters
        model = lgb.LGBMRegressor(
            # objective="regression",
            importance_type='gain',
            ** best_params,
            random_state=42,
            verbosity=-1
        )

        model.fit(X, y)

        return model


class LGBMRollingRanker(TransformerMixin):

    def __init__(self) -> None:
        super(LGBMRollingRanker, self).__init__()
        # pass

    def objective(self, trial, X, y):

        # 参数网格 - 针对股票期货排名预测优化
        params = {
            "boosting_type": 'gbdt',  # 梯度提升决策树，适合排序任务
            'objective': 'lambdarank',  # 排序优化目标
            'eval_metric': 'ndcg',  # 使用NDCG评估排序质量
            'eval_at': [5, 10, 20],  # 评估前5、10、20个预测结果

            # 树的结构参数
            # 控制树的复杂度，金融市场数据通常需要适度复杂的模型捕捉非线性关系
            "num_leaves": trial.suggest_int("num_leaves", 20, 60, step=5),  # 增加上限以捕捉更复杂的模式
            "max_depth": trial.suggest_int("max_depth", 4, 8, 1),  # 适当增加深度范围以捕捉市场的复杂关系

            # 样本和特征采样参数
            # 金融时序数据需要更谨慎的样本处理
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 150, step=10),  # 降低下限以增加灵活性
            # 样本采样 - 金融数据中可能存在异常值，适当降低采样率
            "subsample": trial.suggest_float("subsample", 0.6, 0.9, step=0.05),
            # 特征采样 - 允许更广泛的特征组合探索
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9, step=0.05),
            
            # 训练参数
            # 金融排序任务需要足够的树来捕捉复杂的市场模式
            "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=50),  # 增加树的数量上限
            # 使用较小的学习率以提高稳定性和泛化能力
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, step=0.005),

            # 正则化参数 - 金融数据容易过拟合，需要适当的正则化
            # L1正则化 - 有助于特征选择
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1e-1),
            # L2正则化 - 控制模型复杂度
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 1e-1),

            # 特征处理参数
            "max_bin": trial.suggest_int("max_bin", 200, 350, step=25),  # 增加上限以更精细地处理连续特征
            
            # 新增参数，适合金融时序数据
            "feature_fraction_seed": 42,  # 确保特征采样的可重复性
            "bagging_seed": 42,  # 确保样本采样的可重复性
            "early_stopping_round": 50,  # 防止过拟合的早停策略
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),  # 控制样本采样频率
            
            # 其他参数
            "random_state": 42,
            "verbosity": -1
        }

        tscv = TimeSeriesSplit(n_splits=5)
        ndcg_scores = []

        for idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # train data
            train_data = lgb.Dataset(X_train,
                                     y_train,
                                     group=y_train.groupby(
                                         level=0).size().tolist()
                                     )

            # val data
            val_data = lgb.Dataset(X_val, y_val,
                                   group=y_val.groupby(level=0).size().tolist()
                                   )

            model = lgb.train(params, train_data, valid_sets=[val_data])

            # 计算多个位置的NDCG分数
            ndcg_5 = model.best_score['valid_0']['ndcg@5']
            ndcg_10 = model.best_score['valid_0']['ndcg@10']
            ndcg_20 = model.best_score['valid_0']['ndcg@20']

            # 加权平均不同位置的NDCG
            weighted_ndcg = (0.5 * ndcg_5 + 0.3 * ndcg_10 + 0.2 * ndcg_20)
            ndcg_scores.append(weighted_ndcg)

        return -np.mean(ndcg_scores)  # 返回负值因为Optuna默认最小化目标

    def optimizer(self, study_name="lgb_rolling_ranker", X=None, y=None):

        # Create the Optuna study
        # storage_name = "sqlite:///{}.db".format(study_name)
        storage_name = Config.get_postgresql_uri()

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name,
            direction="minimize",
            storage=storage_name,
            pruner=optuna.pruners.HyperbandPruner(),  # MedianPruner,HyperbandPruner
            load_if_exists=True
        )

        # Optimize the hyperparameters
        study.optimize(lambda trial: self.objective(
            trial, X, y), n_trials=1000)

        # Get the best hyperparameters
        best_params = study.best_params
        print(f'best paras: {study.best_params}')
        print(f'Best NDCG@3 Score: {study.best_value}')

        # Train the LightGBM model with the best hyperparameters
        model = lgb.LGBMRanker(
            boosting_type='gbdt',
            class_weight='balanced',
            importance_type='gain',
            **best_params,
            random_state=42,
            verbosity=-1
        )

        model.fit(X, y, group=y.groupby(level=0).size().tolist())

        return model


class LGBMRollingClassifier(TransformerMixin):
    """LGBM 分类器"""

    def __init__(self) -> None:
        super(LGBMRollingClassifier, self).__init__()

    def objective(self, trial, X, y):
        # 参数网格
        params = {
            # 二分类目标函数和评估指标
            'objective': 'binary',
            'metric': 'binary_logloss',
            # 多分类配置（如需使用请取消注释）
            # 'objective': 'multiclass',
            # 'metric': 'multi_logloss',
            # 'num_class': len(set(y)),  # 设置类别数
            'class_weight': 'balanced',  # 处理不平衡数据
            'importance_type': 'gain',
            
            # 核心超参数
            "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),  # 添加dart提高稳定性
            "num_leaves": trial.suggest_int("num_leaves", 20, 150, step=10),  # 取消注释并扩大范围
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),  # 扩大范围以找到更优解
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),  # 使用对数尺度搜索
            "max_depth": trial.suggest_int("max_depth", 3, 12, 1),  # 略微扩大深度范围
            
            # 防止过拟合参数
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, 5),  # 扩大范围
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),  # 调整下限，金融数据通常需要更多样本
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),  # 特征抽样，减少过拟合
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),  # L1正则化，使用对数尺度
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),  # L2正则化，使用对数尺度
            
            # 金融时序数据特定参数
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0, step=0.05),  # 每次迭代随机选择特征比例
            "verbosity": -1  # 静默模式
        }
        
        # 注: 对于这个参数空间的复杂度(约10个超参数，多为连续值)，
        # 建议使用300-500个trials较为合适。1000个trials可能会过度搜索，
        # 而且收益递减。如果计算资源充足，可以从300开始，
        # 观察收敛情况后再决定是否增加。

        # 滚动窗口训练和预测
        tscv = TimeSeriesSplit(n_splits=5)

        cv_scores = np.empty(5)
        for idx, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            # 创建LightGBM训练集和验证集
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

            # 训练模型
            model = lgb.train(params, train_data, valid_sets=[
                              train_data, val_data])

            # 预测验证集
            # y_pred = np.argmax(model.predict(X_val), axis=1)
            # 预测验证集 (二分类使用概率阈值 0.5)
            y_prob = model.predict(X_val)
            y_pred = (y_prob >= 0.5).astype(int)

            # 计算准确率作为目标函数的评估指标
            cv_scores[idx] = accuracy_score(y_val, y_pred)

        return np.mean(cv_scores)

    def optimizer(self, study_name: str = "lgb_rolling_clf", X=None, y=None):

        # Create the Optuna study
        # storage_name = "sqlite:///{}.db".format(study_name)

        storage_name = Config.get_postgresql_uri()

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name,
            direction="maximize",
            storage=storage_name,
            pruner=optuna.pruners.HyperbandPruner(),
            load_if_exists=True
        )

        # Optimize the hyperparameters
        study.optimize(lambda trial: self.objective(
            trial, X, y), n_trials=1000)

        # Get the best hyperparameters
        best_params = study.best_params
        print(f'best params: {study.best_params}')
        print(f'best value: {study.best_value}')

        # Train the LightGBM model with the best hyperparameters
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            # num_class=len(set(y)),  # 设置类别数量
            class_weight='balanced',
            importance_type='gain',
            **best_params,
            random_state=42,
            verbosity=-1
        )

        model.fit(X, y)

        return model


class TabnetRollReg(TransformerMixin):

    def __init__(self) -> None:
        # super().__init__()
        pass

    def objective(self, trial, X, y):

        # 定义超参空间

        params = {
            'n_d': trial.suggest_int('n_d', 8, 64, 8),
            'n_a': trial.suggest_int('n_a', 8, 64, 8),
            'n_steps': trial.suggest_int('n_steps', 3, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0, step=0.2),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-1),
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=trial.suggest_loguniform('lr', 1e-3, 1e-1)),
            'mask_type': 'sparsemax',
            'seed': 42,
            'verbose': 1
        }

        # 初始化 TabNet 模型
        model = TabNetRegressor(**params)

        # 滚动窗口训练和预测
        tscv = TimeSeriesSplit(n_splits=5)

        # 计算交叉验证指标
        cv_scores = np.empty(5)
        for idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
            y_train, y_val = y.iloc[train_idx].values.reshape(
                -1, 1), y.iloc[val_idx].values.reshape(-1, 1)

            model.fit(
                X_train, y_train,
                max_epochs=100,
                patience=10,
                batch_size=256,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=True  # 确保最后一个批次被丢弃，以防止索引超出范围
            )

            y_pred = model.predict(X_val)
            # val_loss = mean_squared_error(y_val.values, preds)
            # val_losses.append(val_loss)

            cv_scores[idx] = np.sum(
                np.where((y_val < y_pred), 1, 0))/len(y_pred)

        # 返回交叉验证的平均损失作为优化目标
        return np.mean(cv_scores)

    def optimizer(self, study_name: str = "tabnet_rolling_reg", X=None, y=None):

        # 创建Optuna研究
        storage_name = f'postgresql://postgres:035115@localhost:5432/optuna'

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name,
            direction="maximize",
            storage=storage_name,
            pruner=optuna.pruners.HyperbandPruner(),
            load_if_exists=True
        )

        # 优化超参数
        try:
            study.optimize(lambda trial: self.objective(
                trial, X, y), n_trials=50)
        except Exception as e:
            print(f"优化过程中出现错误: {e}")
            return None
        # 获取最佳超参数
        best_params = study.best_params
        print(f'best params: {study.best_params}')
        print(f'best value: {study.best_value}')

        model = TabNetRegressor(**best_params)

        return model
