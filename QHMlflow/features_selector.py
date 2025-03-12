import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List

from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    SequentialFeatureSelector
)

from sklearn.base import TransformerMixin,clone
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# 基于互信息
class MIFeatureSelector(TransformerMixin):
    """基于互信息的特征选择器
    
    Args:
        model_type (str): 模型类型，'reg' 用于回归，'clf' 用于分类
        n (int): 要选择的特征数量
    """
    def __init__(self, model_type: str = 'reg', n: int = 5) -> None:
        super().__init__()
        if model_type not in ['reg', 'clf']:
            raise ValueError("model_type must be either 'reg' or 'clf'")
        if n < 1:
            raise ValueError("n must be positive")
            
        self.model_type = model_type
        self.n = n
        self.selected_features_ = None  # 存储选择的特征名称

    def fit(self, X, y):
        """拟合方法（为保持接口一致）"""
        return self

    def transform(self, X, y):
        """执行特征选择转换
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y (pd.Series): 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.model_type == 'reg':
            selected_features_mi = mutual_info_regression(X, y, random_state=42)
        else:  # clf
            selected_features_mi = mutual_info_classif(X, y, random_state=42)

        # 计算特征重要性并排序
        feature_importance = dict(zip(X.columns, selected_features_mi))
        self.selected_features_ = [k for k, v in sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.n]]

        return X.loc[:, self.selected_features_]


# 基于斯皮尔曼相关
class CorrFeatureSelector(TransformerMixin):
    """基于斯皮尔曼相关系数的特征选择器
    
    Args:
        n (int): 要选择的特征数量
    """
    def __init__(self, n: int = 10) -> None:
        super(CorrFeatureSelector, self).__init__()
        
        if n < 1:
            raise ValueError("n must be positive")
        self.n = n
        self.cols_to_keep = None  # 存储选择的特征名称

    def fit(self, X, y):
        """拟合方法（为保持接口一致）"""
        return self

    def transform(self, X, y):
        """执行特征选择转换
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y (pd.Series): 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array")

        # 确保y是Series类型
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)

        # 创建一个新的DataFrame存放特征值和目标值
        try:
            pm = pd.concat([X, y], axis=1)
        except Exception as e:
            raise ValueError(f"Error merging X and y: {str(e)}")

        # 计算斯皮尔曼相关系数
        corr_series = pm.corr(method='spearman')[pm.columns[-1]].abs()
        self.cols_to_keep = corr_series.nlargest(self.n + 1)[1:].index.tolist()  # 排除目标变量

        # 只保留特征值
        self.cols_to_keep = [c for c in self.cols_to_keep if c in X.columns]

        return X[self.cols_to_keep]


# 基于决策树
class DTFeatureSelector(TransformerMixin):
    """基于决策树的特征选择器
    
    Args:
        n (int): 要选择的特征数量
        model_type (str): 模型类型，'reg' 用于回归，'clf' 用于分类
        random_state (int, optional): 随机种子. Defaults to 42.
    """
    def __init__(self, n: int = 5, model_type: str = 'clf', random_state: int = 42) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("n must be positive")
        if model_type not in ['reg', 'clf']:
            raise ValueError("model_type must be either 'reg' or 'clf'")
        
        self.n = n
        self.model_type = model_type
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """执行特征选择转换
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y: 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # 训练决策树模型
        if self.model_type == 'clf':
            model = DecisionTreeClassifier(random_state=self.random_state)
        else:  # 'reg'
            model = DecisionTreeRegressor(random_state=self.random_state)

        try:
            model.fit(X, y)
        except Exception as e:
            raise ValueError(f"Error fitting decision tree: {str(e)}")

        # 计算并存储特征重要性
        self.feature_importances_ = dict(zip(X.columns, model.feature_importances_))
        self.selected_features_ = [k for k, v in sorted(
            self.feature_importances_.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.n]]

        return X.loc[:, self.selected_features_]


# 基于LGBM树模型特征选择
class LGBMFeatureSelector(TransformerMixin):
    """基于LightGBM的特征选择器
    
    Args:
        model_type (str): 模型类型，'reg' 用于回归，'clf' 用于分类
        itype (str): 特征重要性类型，可选 'gain', 'split'
        n (int): 要选择的特征数量
        random_state (int, optional): 随机种子. Defaults to 42.
    """
    def __init__(self, model_type: str = 'reg', itype: str = 'gain', 
                 n: int = 5, random_state: int = 42) -> None:
        super().__init__()
        if model_type not in ['reg', 'clf']:
            raise ValueError("model_type must be either 'reg' or 'clf'")
        if itype not in ['gain', 'split']:
            raise ValueError("itype must be either 'gain' or 'split'")
        if n < 1:
            raise ValueError("n must be positive")
            
        self.model_type = model_type
        self.itype = itype
        self.n = n
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """拟合方法（为保持接口一致）"""
        return self

    def transform(self, X, y):
        """执行特征选择转换
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y: 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # 配置模型参数
        params = {
            'importance_type': self.itype,
            'n_estimators': 1000,
            'random_state': self.random_state,
            'verbosity': -1
        }
        
        # 根据模型类型选择合适的模型
        if self.model_type == 'reg':
            model = lgb.LGBMRegressor(**params)
        else:  # clf
            model = lgb.LGBMClassifier(class_weight='balanced', **params)

        # 训练模型并处理可能的异常
        try:
            model.fit(X, y)
        except Exception as e:
            raise ValueError(f"Error fitting LightGBM model: {str(e)}")

        # 计算并存储特征重要性
        self.feature_importances_ = dict(zip(model.feature_name_, model.feature_importances_))
        self.selected_features_ = [k for k, v in sorted(
            self.feature_importances_.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.n]]

        return X.loc[:, self.selected_features_]


# 基于F值特征选择
class FValueFeatureSelector(TransformerMixin):
    """基于F值的特征选择器
    
    Args:
        model_type (str): 模型类型，'reg' 用于回归，'clf' 用于分类
        n (int): 要选择的特征数量
    """
    def __init__(self, model_type: str = 'reg', n: int = 10) -> None:
        super().__init__()
        if model_type not in ['reg', 'clf']:
            raise ValueError("model_type must be either 'reg' or 'clf'")
        if n < 1:
            raise ValueError("n must be positive")
            
        self.model_type = model_type
        self.n = n
        self.selected_features_ = None  # 存储选择的特征名称

    def fit(self, X, y):
        """拟合方法（为保持接口一致）"""
        return self

    def transform(self, X, y):
        """执行特征选择转换
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y (pd.Series): 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array")

        # 创建选择器
        if self.model_type == 'reg':
            selector = SelectKBest(score_func=f_regression, k=self.n)
        elif self.model_type == 'clf':
            selector = SelectKBest(score_func=f_classif, k=self.n)

        # 拟合选择器
        try:
            selector.fit(X, y)
        except Exception as e:
            raise ValueError(f"Error fitting SelectKBest: {str(e)}")

        # 获取选择的特征
        self.selected_features_ = selector.get_feature_names_out()

        return X[self.selected_features_]


# 基于随机森林
class RFFeatureSelector(TransformerMixin):
    """基于随机森林的特征选择器
    
    Args:
        n (int): 要选择的特征数量
        model_type (str): 模型类型，'reg' 用于回归，'clf' 用于分类
        random_state (int, optional): 随机种子. Defaults to 42.
    """
    def __init__(self, n: int = 5, model_type: str = 'clf', random_state: int = 42) -> None:
        super().__init__()
        if model_type not in ['reg', 'clf']:
            raise ValueError("model_type must be either 'reg' or 'clf'")
        self.n = n
        self.model_type = model_type
        self.random_state = random_state
        self.selected_features_ = None

    def transform(self, X, y):
        """执行特征选择转换
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y: 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.model_type == 'clf':
            model = RandomForestClassifier(
                n_estimators=500, 
                random_state=self.random_state
            )
        elif self.model_type == 'reg':
            model = RandomForestRegressor(
                n_estimators=500, 
                random_state=self.random_state
            )
        else:
            raise ValueError("Invalid model_type. Must be 'reg' or 'clf'.")

        model.fit(X, y)

        feature_importance_dict = dict(
            zip(X.columns, model.feature_importances_)
        )
        self.selected_features_ = [k for k, v in sorted(
            feature_importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.n]]

        return X[self.selected_features_]


# 递归特征消除（Recursive Feature Elimination, RFE）
class RFEFeaturesSelector(TransformerMixin):
    """基于递归特征消除的特征选择器
    
    Args:
        base_model: 基础模型
        n (int): 要选择的特征数量
    """
    def __init__(self, base_model, n: int = 10) -> None:
        super().__init__()
        self.base_model = base_model
        self.n = n
        self.selected_features_ = None

    def transform(self, X, y):
        """执行特征选择转换"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        selector = RFE(
            estimator=clone(self.base_model),
            n_features_to_select=self.n, 
            step=10
        )
        selector.fit(X, y)
        self.selected_features_ = X.columns[selector.get_support()].tolist()
        
        return X[self.selected_features_]


# 顺序特征选择（Sequential Feature Selection）
class SFSFeaturesSelector(TransformerMixin):

    def __init__(self, base_model, n: int = 10) -> None:
        self.base_model = base_model  # 基础模型
        self.n = n
        self.selector = None

    def fit(self, X, y):

        return self

    def transform(self, X, y):

        # 创建顺序特征选择器
        selector = SequentialFeatureSelector(
            self.base_model, n_features_to_select=self.n, direction='backward')
        # 拟合选择器
        selector.fit(X, y)

        # 获取选择的特征
        selected_features = selector.get_support(indices=True)

        # 返回选择特征后的DataFrame
        return X.iloc[:, selected_features]


# 选择最佳选择器
class BestFeaturesSelector(TransformerMixin):
    def __init__(self, base_model, model_type: str = 'reg', n_features: List[int] = [10, 15, 20, 25, 30]) -> None:
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        self.n_features = n_features
        self.best_method_ = None
        self.best_features_ = None
        self.scores_ = None

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """执行特征选择并返回最佳特征集
        
        Args:
            X (pd.DataFrame): 输入特征矩阵
            y: 目标变量
            
        Returns:
            pd.DataFrame: 选择后的特征矩阵
        """
        scores = {}
        best_features = {}
        
        for n in self.n_features:
            methods = {
                'LGBM': LGBMFeatureSelector(model_type=self.model_type, n=n),
                'F-value': FValueFeatureSelector(model_type=self.model_type, n=n),
                'MI': MIFeatureSelector(model_type=self.model_type, n=n),
                'RF': RFFeatureSelector(n=n)
            }

            for name, method in methods.items():
                try:
                    X_selected = method.transform(X, y)
                    model = clone(self.base_model)  # 克隆模型避免状态污染
                    model.fit(X_selected, y)
                    y_pred = model.predict(X_selected)
                    
                    method_name = f'{name}({n})'
                    score = mean_squared_error(y, y_pred) if self.model_type == 'reg' else accuracy_score(y, y_pred)
                    scores[method_name] = score
                    best_features[method_name] = X_selected.columns.tolist()
                except Exception as e:
                    print(f"Error in method {name} with {n} features: {str(e)}")
                    continue

        # 根据模型类型选择最佳方法
        self.scores_ = dict(sorted(scores.items(), key=lambda x: x[1], reverse=self.model_type=='clf'))
        self.best_method_ = list(self.scores_.keys())[0]
        self.best_features_ = best_features[self.best_method_]
        
        return X[self.best_features_]
