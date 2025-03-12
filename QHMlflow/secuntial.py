import os
import uuid
import time
import pickle
import warnings
import pandas as pd
import numpy as np
import json
from typing import List

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from QHUtil.loggers import watch_dog
from QHMlflow.mlflowtracker import log_mlflow
from QHMlflow.evaluator import performance_indicators
from QHData.data_load import (DataLoader, CodesLoader)
from QHFactor.factors_calclatror import FactorsBatchCalculator
from QHFactor.transformer import (
    Load_X_y,
    RollingZScoreTransformer,
    RollingWinsorizeTransformer
)

warnings.filterwarnings('ignore')


class Sequential:
    """机器学习流水线序列类

    用于构建完整的机器学习工作流，包括数据加载、特征计算、
    特征选择、模型训练、回测和模型追踪等步骤。

    Attributes:
        name (str): 模型名称
        logger: 日志记录器实例
    """

    def __init__(self, name: str = None) -> None:
        self.logger = watch_dog(filename='mlflow.log')
        self.name = name or f"model_{uuid.uuid4().hex[:8]}"

        self.logger.info(f'Model name: {self.name}')
        self.logger.info(
            f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def data_loader(self, start_date: str, end_date: str, freq: str, codes: List = []):
        """
        Load dataset based on specified frequency and date range.

        Parameters:
        start_date (str): Start date of the data to load.
        end_date (str): End date of the data to load.
        freq (str): Frequency of the data ('d', 'D', '60min', '30min', '15min', '5min', '1min').
        codes (List, optional): List of codes to load. If not provided, defaults will be used.

        Returns:
        tuple: Loaded dataset and returns.
        """
        self.logger.info('# 1. Get dataset.')
        self.logger.info('Data loading...')

        def _logger(dataset, codes):
            self.logger.info(f'Codes: {codes}')
            self.logger.info(f'Dataset from {start_date} to {end_date}')
            self.logger.info(f'Frequency: {freq}')
            self.logger.info(f'Dataset shape: {dataset.shape}')
            self.logger.info('Data loading completed')

        def is_valid_frequency(freq):
            valid_frequencies = ['d', 'D', '60min',
                                 '30min', '15min', '5min', '1min']
            return freq in valid_frequencies

        if not is_valid_frequency(freq):
            self.logger.error('Invalid frequency specified.')
            return None, None

        if not codes:
            code_loader_n = 30 if freq in ['d', 'D'] else 10
            codes = CodesLoader(n=code_loader_n).load_fut_codes()

        try:
            dataset, returns = DataLoader(start_date, end_date, freq=freq).load_future(
                codes=codes, return_X_y=True)
        except Exception as e:
            self.logger.error(f'Error loading dataset: {e}')
            return None, None

        if dataset is not None and returns is not None:
            _logger(dataset, codes)
            return dataset, returns
        else:
            self.logger.error('Failed to load dataset.')
            return None, None

    def calculate_factors(self, dataset: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算因子并进行处理。

        参数:
        - dataset: pd.DataFrame, 输入的数据集
        - window: int, 滚动窗口的大小,默认为20

        返回:
        - factors: pd.DataFrame, 处理后的因子数据
        """
        self.logger.info('# 2. calculate factors')
        self.logger.info('Factor calculation and processing started')

        # 定义处理流水线
        processor = Pipeline([
            ('calculate', FactorsBatchCalculator()),        # 批量计算因子
            ('roll_winsorize', RollingWinsorizeTransformer(window=window)),  # 滚动去极值处理
            ('roll_zscore', RollingZScoreTransformer(window=window))  # 滚动标准化处理
        ])

        try:
            # 进行因子计算和处理
            factors = processor.fit_transform(X=dataset)
            self.logger.info('Factors calculated.')
            self.logger.info(f'Factors shape: {factors.shape}')

            # 记录因子数量和类型
            factor_types = factors.dtypes.to_dict()
            self.logger.info(f'Number of factors: {factors.shape[1]}')
            self.logger.info(f'Factor types: {factor_types}')

        except Exception as e:
            self.logger.error(f'Error during factor calculation: {e}')
            raise

        return factors

    def select_features(self, selector, X, y):
        """
        Selects features using a provided selector and logs the process.

        Parameters:
        selector: The feature selector object implementing a transform method.
        X: DataFrame or array-like, shape (n_samples, n_features)
            The input samples.
        y: array-like, shape (n_samples,)
            The target values.

        Returns:
        selected_factors: DataFrame
            The selected features.
        """
        self.logger.info('# 3. Selecting features')

        # 验证 selector 是否具有 transform 方法
        if not hasattr(selector, 'transform'):
            self.logger.error('Selector does not have a transform method.')
            raise AttributeError('Selector does not have a transform method.')

        # 确保 X 和 y 被正确转换
        try:
            self.logger.info('Transforming X and y...')
            feature, labels = Load_X_y().transform(X, y)
            self.logger.info('Transformation completed.')
        except Exception as e:
            self.logger.error(f'Error in loading X and y: {e}')
            raise

        # 应用特征选择
        try:
            self.logger.info('Applying feature selection...')
            selected_factors = selector.transform(feature, labels)
            self.logger.info('Feature selection completed.')
        except Exception as e:
            self.logger.error(f'Error in feature selection: {e}')
            raise

        self.logger.info(f'Selected features shape: {selected_factors.shape}')

        # 绘制所选特征的相关性
        self.logger.info('Plotting correlation of selected features...')
        try:
            # plot_correlation(selected_factors)
            self.plot_feature_correlation(features=selected_factors)
            self.logger.info('Correlation plot completed.')
        except Exception as e:
            self.logger.error(f'Error in plotting correlation: {e}')
            raise

        # 记录所选因子的名称
        self.logger.info(
            f'Factor selection: {selected_factors.columns.tolist()}')

        return selected_factors

    def train_model(self, algo, X_train: pd.DataFrame, y_train: pd.Series):
        """训练模型

        Args:
            algo: 算法类，需要实现 optimizer 方法
            X_train (pd.DataFrame): 训练特征数据
            y_train (pd.Series): 训练标签数据

        Returns:
            训练好的模型实例

        Raises:
            ValueError: 当算法不符合要求或训练失败时
        """
        self.logger.info('# 4. Training model')

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame")

        if not isinstance(y_train, (pd.Series, np.ndarray)):
            raise TypeError("y_train must be a pandas Series or numpy array")

        try:
            # 验证算法类
            algorithm = algo()
            if not hasattr(algorithm, 'optimizer'):
                raise ValueError("Algorithm must implement optimizer method")

            # 训练模型
            study_name = f'{self.name}_{uuid.uuid4().hex[:8]}'
            self.logger.info(
                f'Starting optimization with study name: {study_name}')

            model = algorithm.optimizer(
                study_name=study_name,
                X=X_train,
                y=y_train
            )

            self.logger.info('Model training completed successfully')
            return model

        except Exception as e:
            self.logger.error(f'Error during model training: {e}')
            raise

    def track_model(self,
                    exp_name: str,
                    model,
                    params: dict = None,
                    metrics: dict = None,
                    tags: dict = None,
                    image: str = None,
                    config=None,
                    track_ui: bool = True) -> None:
        """
        使用MLflow跟踪模型训练结果

        Parameters:
        -----------
        exp_name : str
            实验名称
        model : object
            训练好的模型对象
        params : dict, optional
            模型参数字典
        metrics : dict, optional
            模型评估指标字典

        tags : dict, optional
            标签信息
        artifacts : dict, optional
            需要保存的文件路径字典
        image : str, optional
            性能图表本地路径
        track_ui : bool, default True
            是否在UI中显示跟踪结果

        Raises:
        -------
        ValueError
            当必要参数缺失或格式不正确时
        Exception
            当MLflow跟踪过程中出现错误时
        """
        try:
            self.logger.info('Starting model tracking with MLflow...')

            # 验证必要参数
            if not exp_name:
                raise ValueError("Experiment name cannot be empty")
            if model is None:
                raise ValueError("Model cannot be None")

            # 验证可选参数格式
            if params is not None and not isinstance(params, dict):
                raise ValueError("params must be a dictionary")
            if metrics is not None and not isinstance(metrics, dict):
                raise ValueError("metrics must be a dictionary")
            if tags is not None and not isinstance(tags, dict):
                raise ValueError("tags must be a dictionary")
            # if artifacts is not None and not isinstance(artifacts, dict):
            #     raise ValueError("artifacts must be a dictionary")

            # 验证图片路径
            if image and not os.path.exists(image):
                self.logger.warning(f"Performance image not found at: {image}")

            # 准备跟踪参数
            tracking_params = {
                'experiment_name': self.name,
                'model': model,
                'params': params or {},
                'metrics': metrics or {},
                'tags': tags or {},
                # 'artifacts': artifacts or {},
                "config": config,
                'image_local_path': image,
                'track_ui': track_ui
            }

            # 记录跟踪信息
            self.logger.info(f"Tracking experiment: {exp_name}")
            if params:
                self.logger.info(f"Parameters to track: {list(params.keys())}")
            if metrics:
                self.logger.info(f"Metrics to track: {list(metrics.keys())}")
            # if artifacts:
            #     self.logger.info(
            #         f"Artifacts to track: {list(artifacts.keys())}")

            # 执行MLflow跟踪
            run_id = log_mlflow(**tracking_params)    
            
            self.logger.info(
                f'Model "{self.name}" successfully tracked in MLflow')
            
            return run_id
        
        except ValueError as ve:
            self.logger.error(f'Validation error in track_model: {ve}')
            raise
        except Exception as e:
            self.logger.error(f'Error tracking model in MLflow: {e}')
            raise

    def save_model(self, model, model_path):
        """Save model to local storage."""
        try:
            # 检查路径是否有效
            if not model_path:
                raise ValueError("Model path is not specified.")

            # 尝试保存模型
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f'{self.name} model saved to {model_path}')
        except Exception as e:
            self.logger.error(f'Error saving model: {e}')
            raise

    def load_model(self, model_path):
        """Load model from local storage."""
        try:
            # 检查路径是否有效
            if not model_path:
                raise ValueError("Model path is not specified.")

            # 尝试加载模型
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f'{self.name} model loaded from {model_path}')
            return model
        except FileNotFoundError:
            self.logger.error(f'Model file not found at {model_path}')
            raise
        except Exception as e:
            self.logger.error(f'Error loading model: {e}')
            raise

    def plot_feature_importance(self, model, features: pd.DataFrame, save_path: str = None):
        """绘制特征重要性图

        Args:
            model: 训练好的模型实例
            features (pd.DataFrame): 特征数据
            save_path (str, optional): 图片保存路径
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(
                    model.feature_importances_,
                    index=features.columns
                ).sort_values(ascending=True)

                plt.figure(figsize=(10, 6))
                importances.plot(kind='barh')
                plt.title('Feature Importance')

                if save_path:
                    plt.savefig(save_path)
                    self.logger.info(
                        f'Feature importance plot saved to {save_path}')

                plt.close()

        except Exception as e:
            self.logger.error(f'Error plotting feature importance: {e}')

    def plot_feature_correlation(self, features: pd.DataFrame,
                                 save_path: str = None,
                                 figsize: tuple = (12, 10),
                                 method: str = 'spearman',
                                 cmap: str = 'coolwarm',
                                 annot: bool = True,
                                 fmt: str = '.2f') -> None:
        """绘制特征相关性热力图

        Args:
            features (pd.DataFrame): 特征数据框
            save_path (str, optional): 图片保存路径
            figsize (tuple, optional): 图形大小. Defaults to (12, 10).
            method (str, optional): 相关系数计算方法 {'pearson', 'kendall', 'spearman'}. 
                Defaults to 'spearman'.
            cmap (str, optional): 热力图配色方案. Defaults to 'coolwarm'.
            annot (bool, optional): 是否显示相关系数值. Defaults to True.
            fmt (str, optional): 相关系数值的格式. Defaults to '.2f'.

        Raises:
            TypeError: 当输入数据类型不正确时
            ValueError: 当相关系数计算方法不支持时
        """
        try:
            self.logger.info('Plotting feature correlation matrix...')

            # 参数验证
            if not isinstance(features, pd.DataFrame):
                raise TypeError("features must be a pandas DataFrame")

            if method not in ['pearson', 'kendall', 'spearman']:
                raise ValueError(
                    "method must be one of 'pearson', 'kendall', 'spearman'")

            # 计算相关系数矩阵
            corr_matrix = features.corr(method=method)

            # 创建图形
            plt.figure(figsize=figsize)

            # 使用seaborn绘制热力图
            import seaborn as sns
            mask = np.triu(np.ones_like(corr_matrix), k=1)  # 创建上三角掩码

            sns.heatmap(corr_matrix,
                        #    mask=mask,
                        cmap=cmap,
                        annot=annot,
                        fmt=fmt,
                        square=True,
                        linewidths=0.5,
                        cbar_kws={"shrink": .5})

            plt.title(f'Feature Correlation Matrix ({method})')
            plt.tight_layout()

            plt.show()
            # 保存图片
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f'Correlation plot saved to {save_path}')

            plt.close()

            # # 输出高相关性警告
            # high_corr_pairs = self._get_high_correlation_pairs(corr_matrix, threshold=0.8)
            # if high_corr_pairs:
            #     # self.logger.warning('High correlation pairs detected (>0.8):')
            #     for pair, corr in high_corr_pairs:
            #         self.logger.warning(f'{pair}: {corr:.3f}')

        except Exception as e:
            self.logger.error(f'Error plotting correlation matrix: {e}')
            raise

    def _get_high_correlation_pairs(self, corr_matrix: pd.DataFrame,
                                    threshold: float = 0.8) -> List[tuple]:
        """获取高相关性特征对

        Args:
            corr_matrix (pd.DataFrame): 相关系数矩阵
            threshold (float, optional): 相关性阈值. Defaults to 0.8.

        Returns:
            List[tuple]: 高相关性特征对列表，每个元素为 ((feature1, feature2), correlation)
        """
        high_corr = []

        # 获取上三角矩阵的高相关性对
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append(
                        ((corr_matrix.columns[i], corr_matrix.columns[j]),
                         corr_matrix.iloc[i, j])
                    )

        return sorted(high_corr, key=lambda x: abs(x[1]), reverse=True)

    def update_run_ids(self, model_name: str, run_id: str, file_path: str = 'run_ids.json'):
        """更新run_ids配置文件

        Args:
            model_name (str): 模型名称
            run_id (str): MLflow运行ID
            file_path (str): JSON文件路径，默认为'run_ids.json'
        """
        # 如果文件存在则读取，否则创建空字典
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                run_ids = json.load(f)
        else:
            run_ids = {}

        # 更新run_id
        run_ids[model_name] = run_id

        # 写入文件
        with open(file_path, 'w') as f:
            json.dump(run_ids, f, indent=4)

    def load_run_ids(self, file_path: str = 'run_ids.json') -> dict:
        """加载run_ids配置文件

        Args:
            file_path (str): JSON文件路径，默认为'run_ids.json'

        Returns:
            dict: 包含模型名称和run_id的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Run IDs file not found at {file_path}")

        with open(file_path, 'r') as f:
            return json.load(f)
