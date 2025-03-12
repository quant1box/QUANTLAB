import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FactorSelector(TransformerMixin):
    """因子選擇器"""
    
    def __init__(self,
                 method: str = 'ic',
                 ic_threshold: float = 0.02,
                 pvalue_threshold: float = 0.05,
                 corr_threshold: float = 0.7):
        """
        參數:
            method: 選擇方法 'ic'/'pca'/'clustering'
            ic_threshold: IC閾值
            pvalue_threshold: p值閾值
            corr_threshold: 相關係數閾值
        """
        self.method = method
        self.ic_threshold = ic_threshold
        self.pvalue_threshold = pvalue_threshold
        self.corr_threshold = corr_threshold
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FactorSelector':
        """
        擬合選擇器
        
        參數:
            X: 因子值DataFrame
            y: 目標變量
        """
        if self.method == 'ic':
            self.selected_factors_ = self._select_by_ic(X, y)
        elif self.method == 'pca':
            self.selected_factors_ = self._select_by_pca(X)
        elif self.method == 'clustering':
            self.selected_factors_ = self._select_by_clustering(X)
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        轉換數據
        
        參數:
            X: 因子值DataFrame
            
        返回:
            選中的因子DataFrame
        """
        return X[self.selected_factors_]
    
    def _select_by_ic(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """基於IC選擇因子"""
        selected = []
        
        for col in X.columns:
            ic = X[col].corr(y)
            if abs(ic) > self.ic_threshold:
                selected.append(col)
                
        return selected
    
    def _select_by_pca(self, X: pd.DataFrame) -> List[str]:
        """基於PCA選擇因子"""
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA分析
        pca = PCA()
        pca.fit(X_scaled)
        
        # 選擇解釋90%方差的成分
        n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9) + 1
        
        # 獲取貢獻最大的因子
        component_df = pd.DataFrame(
            pca.components_[:n_components],
            columns=X.columns
        )
        
        selected = []
        for i in range(n_components):
            max_factor = component_df.iloc[i].abs().idxmax()
            selected.append(max_factor)
            
        return selected
    
    def _select_by_clustering(self, X: pd.DataFrame) -> List[str]:
        """基於聚類選擇因子"""
        # 計算相關係數矩陣
        corr_matrix = X.corr().abs()
        
        # 初始化選中的因子列表
        selected = [X.columns[0]]  # 從第一個因子開始
        
        # 遍歷所有因子
        for factor in X.columns[1:]:
            # 檢查與已選因子的相關係數
            max_corr = corr_matrix.loc[factor, selected].max()
            
            # 如果相關係數小於閾值，則選中該因子
            if max_corr < self.corr_threshold:
                selected.append(factor)
                
        return selected 