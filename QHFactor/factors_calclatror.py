
import os
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm
from QHFactor.fn import *
from functools import partial

from QHFactor.factors import GTJA_191
from QHFactor.factors_map import gtja_191_map

from sklearn.base import TransformerMixin
from typing import Dict, Union, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


class FactorsBatchCalculator(TransformerMixin):
    def __init__(self,
                 factors: Union[List[str], Tuple[str]] = None,
                 n_jobs: int = -1,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = False,
                 pre_process: Optional[Callable] = None,
                 post_process: Optional[Callable] = None) -> None:
        """
        Args:
            factors: 要计算的因子列表
            n_jobs: 并行进程数，-1表示使用所有CPU核心
            cache_dir: 缓存目录路径
            use_cache: 是否使用缓存
            pre_process: 数据预处理函数
            post_process: 数据后处理函数
        """
        if factors is not None and not isinstance(factors, (list, tuple)):
            raise TypeError("factors must be a list or tuple of strings")

        self.factors = factors
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.pre_process = pre_process
        self.post_process = post_process

        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_path(self, group_name: str) -> str:
        """生成缓存文件路径"""
        return os.path.join(self.cache_dir, f"factors_cache_{group_name}.pkl")

    def _process_group(self, group_data: Tuple[str, pd.DataFrame]) -> pd.DataFrame:
        """处理单个分组的数据"""
        group_name, group = group_data

        # 检查缓存
        if self.use_cache and self.cache_dir:
            cache_path = self._get_cache_path(group_name)
            if os.path.exists(cache_path):
                return joblib.load(cache_path)

        # 预处理
        if self.pre_process:
            group = self.pre_process(group)

        # 计算因子
        gtja = GTJA_191(group)
        factors_values_map = {}

        if not self.factors:
            self.factors = list(gtja_191_map.keys())

        for factor in self.factors:
            if factor in gtja_191_map:
                try:
                    factors_values_map[factor] = gtja_191_map[factor](gtja)
                except Exception as e:
                    print(f"Warning: Failed to calculate factor {factor} for group {group_name}: {str(e)}")
                    factors_values_map[factor] = pd.Series(0, index=group.index)
            else:
                raise ValueError(f"Factor {factor} is not defined.")

        factors = pd.DataFrame(factors_values_map, index=group.index)
        factors = factors.replace([np.inf, -np.inf], 0)

        # 后处理
        if self.post_process:
            factors = self.post_process(factors)

        # 保存缓存
        if self.use_cache and self.cache_dir:
            joblib.dump(factors, self._get_cache_path(group_name))

        return factors

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # 确保factors已初始化
        if not self.factors:
            self.factors = list(gtja_191_map.keys())

        # 获取分组数据
        groups = list(X.groupby(level=1))
        
        # 使用线程池进行并行计算
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # 使用tqdm创建进度条，这次只按股票组数计数
            results = []
            with tqdm(total=len(groups), desc="Processing stocks") as pbar:
                # 提交所有任务
                future_to_group = {
                    executor.submit(self._process_group, group_data): group_data[0]
                    for group_data in groups
                }
                
                for future in as_completed(future_to_group):
                    group_name = future_to_group[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)  # 每完成一个股票组更新一次进度
                    except Exception as e:
                        print(f"Error processing group {group_name}: {str(e)}")
                        pbar.update(1)  # 即使出错也更新进度

        # 合并结果并排序
        return pd.concat(results).sort_index()

    def clear_cache(self):
        """清除所有缓存文件"""
        if self.cache_dir and os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.startswith("factors_cache_"):
                    os.remove(os.path.join(self.cache_dir, file))
