import pandas as pd
import logging
import asyncio
import akshare as ak
from typing import Optional, Dict
from datetime import datetime

class DataFetcher:
    """数据获取器"""
    
    def __init__(self):
        self.cache = {}  # 数据缓存
        self.last_update = None  # 上次更新时间
        
    async def fetch_data(self) -> pd.DataFrame:
        """获取期权市场数据"""
        try:
            current_time = datetime.now()
            
            # 检查缓存是否需要更新 (1分钟更新一次)
            if (self.last_update is None or 
                (current_time - self.last_update).total_seconds() > 60):
                
                # 获取期权数据
                df_value = await self._fetch_value_data()
                df_risk = await self._fetch_risk_data()
                df_premium = await self._fetch_premium_data()
                
                # 合并数据
                merged_data = await self._merge_data(df_value, df_risk, df_premium)


                # 更新缓存
                self.cache = merged_data
                self.last_update = current_time
                
                logging.info(f"数据更新成功: {len(merged_data)}条记录")
                
            return self.cache
            
        except Exception as e:
            logging.error(f"获取数据错误: {e}")
            raise
            
    async def _fetch_value_data(self) -> pd.DataFrame:
        """获取期权价值数据"""
        try:
            df = ak.option_value_analysis_em()
            return self._preprocess_data(df)
        except Exception as e:
            logging.error(f"获取期权价值数据错误: {e}")
            raise
            
    async def _fetch_risk_data(self) -> pd.DataFrame:
        """获取期权风险数据"""
        try:
            df = ak.option_risk_analysis_em()
            return self._preprocess_data(df)
        except Exception as e:
            logging.error(f"获取期权风险数据错误: {e}")
            raise
            
    async def _fetch_premium_data(self) -> pd.DataFrame:
        """获取期权溢价数据"""
        try:
            df = ak.option_premium_analysis_em()
            return self._preprocess_data(df)
        except Exception as e:
            logging.error(f"获取期权溢价数据错误: {e}")
            raise
            
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        # 统一数据类型
        if '期权代码' in df.columns:
            df['期权代码'] = df['期权代码'].astype(str)
        if '到期日' in df.columns:
            df['到期日'] = pd.to_datetime(df['到期日'])
            
        # 处理数值列
        numeric_columns = ['最新价', '时间价值', '内在价值', '隐含波动率', 
                         'Delta', 'Gamma', 'Theta', 'Vega', '行权价']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
        
    async def _merge_data(self, df_value: pd.DataFrame, 
                         df_risk: pd.DataFrame, 
                         df_premium: pd.DataFrame) -> pd.DataFrame:
        """合并数据"""
        try:
            # 合并数据
            merged_df = (df_value
                        .merge(df_risk, on='期权代码', how='left', suffixes=('', '_risk'))
                        .merge(df_premium, on='期权代码', how='left', suffixes=('', '_premium')))
            
            # 添加期权类型
            merged_df['期权类型'] = merged_df['期权名称'].apply(
                lambda x: '认购' if '购' in str(x) else '认沽' if '沽' in str(x) else None
            )
            
            # 计算剩余天数
            merged_df['剩余天数'] = (merged_df['到期日'] - pd.Timestamp.now()).dt.days
            
            return merged_df
            
        except Exception as e:
            logging.error(f"合并数据错误: {e}")
            raise 