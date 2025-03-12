from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Union, Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class FactorMiner:
    """因子挖掘器"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 function_map: Dict[str, Callable],
                 target_returns: pd.Series,
                 population_size: int = 100,
                 generations: int = 50,
                 tournament_size: int = 3,
                 max_depth: int = 3,
                 n_jobs: int = -1):
        """ 
        参数:
            data: 原始数据
            function_map: 算子函数映射
            target_returns: 目标收益率
            population_size: 种群大小
            generations: 迭代代数
            tournament_size: 锦标赛选择大小
            max_depth: 表达式最大深度
            n_jobs: 并行进程数,-1表示使用所有CPU
        """
        self.data = data
        self.function_map = function_map
        self.target_returns = target_returns
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        # 初始化并行池
        self.pool = ThreadPoolExecutor(max_workers=self.n_jobs)
        
        # 初始化遗传算法工具箱
        self._init_toolbox()
        
    def _init_toolbox(self):
        """初始化工具箱"""
        # 创建适应度类和个体类
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # 注册个体生成方法
        self.toolbox.register("expr", self._generate_expr)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册遗传算法操作
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # 启用并行计算
        self.toolbox.register("map", self.pool.map)
        
    def _generate_expr(self) -> List:
        """生成因子表达式"""
        def grow(depth: int) -> Union[str, List]:
            if depth >= self.max_depth:
                # 到达最大深度,返回终端节点
                return random.choice(list(self.data.columns))
            
            # 随机选择一个函数
            func = random.choice(list(self.function_map.keys()))
            func_obj = self.function_map[func]
            
            # 生成参数
            params = []
            for _ in range(func_obj.arity):
                # 递归生成子表达式
                if random.random() < 0.5:  # 50%概率继续生长
                    params.append(grow(depth + 1))
                else:  # 50%概率返回终端节点
                    params.append(random.choice(list(self.data.columns)))
                
            # 如果是时序函数,添加时间窗口参数
            if func_obj.is_ts:
                params.append(random.choice([5, 10, 20, 30]))
                
            return [func] + params
            
        return grow(0)
        
    def _evaluate(self, individual: List) -> tuple:
        """评估个体适应度"""
        try:
            # 计算因子值
            factor_values = self._compute_factor(individual)
            
            # 计算评估指标
            metrics = self._compute_metrics(factor_values)
            
            # 综合评分
            score = self._compute_score(metrics)
            
            return (score,)
            
        except Exception as e:
            return (-1.0,)
            
    def _mutate(self, individual: List) -> tuple:
        """变异操作"""
        if len(individual) < 1:
            return individual,
            
        # 随机选择变异类型
        mutation_type = random.choice(['replace', 'modify', 'insert', 'delete'])
        
        if mutation_type == 'replace':
            # 替换子树
            pos = random.randint(0, len(individual)-1)
            individual[pos:] = self._generate_expr()
            
        elif mutation_type == 'modify':
            # 修改参数
            for i, node in enumerate(individual):
                if isinstance(node, (int, float)):
                    if random.random() < 0.2:
                        if isinstance(node, int):
                            individual[i] = random.choice([5, 10, 20, 30])
                        else:
                            individual[i] *= random.uniform(0.8, 1.2)
                            
        elif mutation_type == 'insert':
            # 插入新节点
            pos = random.randint(0, len(individual))
            individual.insert(pos, self._generate_expr())
            
        else:  # delete
            # 删除节点
            if len(individual) > 1:
                pos = random.randint(0, len(individual)-1)
                individual.pop(pos)
                
        return individual,
        
    def _compute_factor(self, expr: List) -> pd.Series:
        """计算因子值"""
        def evaluate(node):
            if isinstance(node, str):
                if node in self.function_map:
                    return self.function_map[node]
                else:
                    return self.data[node]
            elif isinstance(node, list):
                func = self.function_map[node[0]]
                args = [evaluate(arg) for arg in node[1:]]
                
                if func.is_ts:
                    # 最后一个参数是时间窗口
                    d = args[-1]
                    args = args[:-1]
                    func.set_d(d)
                    
                return func(*args)
            else:
                return node
                
        return evaluate(expr)
        
    def _compute_metrics(self, factor: pd.Series) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 计算IC
        metrics['ic'] = self._compute_ic(factor, self.target_returns)
        
        # 计算IR
        metrics['ir'] = self._compute_ir(factor, self.target_returns)
        
        # 计算因子自相关性
        metrics['autocorr'] = self._compute_autocorr(factor)
        
        # 计算因子收益率
        metrics['returns'] = self._compute_factor_returns(factor)
        
        return metrics
        
    def _compute_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算IC值"""
        return factor.corr(returns)
        
    def _compute_ir(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算IR值"""
        daily_ic = []
        for date in factor.index.unique():
            f = factor.loc[date]
            r = returns.loc[date]
            ic = f.corr(r)
            daily_ic.append(ic)
            
        daily_ic = pd.Series(daily_ic)
        return daily_ic.mean() / daily_ic.std() if daily_ic.std() != 0 else 0
        
    def _compute_autocorr(self, factor: pd.Series, lag: int = 1) -> float:
        """计算因子自相关性"""
        return factor.autocorr(lag)
        
    def _compute_factor_returns(self, factor: pd.Series) -> float:
        """计算因子收益率"""
        # 分组回测
        groups = pd.qcut(factor, 5, labels=False)
        group_returns = pd.DataFrame({
            'group': groups,
            'returns': self.target_returns
        }).groupby('group')['returns'].mean()
        
        # 多空组合收益
        return group_returns.iloc[-1] - group_returns.iloc[0]
        
    def _compute_score(self, metrics: Dict) -> float:
        """计算综合评分"""
        weights = {
            'ic': 0.3,
            'ir': 0.3,
            'autocorr': 0.2,
            'returns': 0.2
        }
        
        score = (
            weights['ic'] * abs(metrics['ic']) +
            weights['ir'] * abs(metrics['ir']) +
            weights['autocorr'] * (1 - abs(metrics['autocorr'])) +
            weights['returns'] * metrics['returns']
        )
        
        return score
        
    def run(self, verbose: bool = True) -> Dict:
        """运行因子挖掘"""
        # 生成初始种群
        pop = self.toolbox.population(n=self.population_size)
        
        # 记录最优解
        hof = tools.HallOfFame(1)
        
        # 记录统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 进度条
        pbar = tqdm(total=self.generations) if verbose else None
        
        def update_pbar(gen, pop, hof):
            if pbar:
                pbar.update(1)
                pbar.set_description(f"Best Fitness: {hof[0].fitness.values[0]:.4f}")
        
        # 运行遗传算法
        pop, log = algorithms.eaSimple(
            pop, 
            self.toolbox,
            cxpb=0.7,  # 交叉概率
            mutpb=0.3,  # 变异概率
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=False,
            callback=update_pbar
        )
        
        if pbar:
            pbar.close()
            
        # 返回最优解
        best_expr = hof[0]
        best_factor = self._compute_factor(best_expr)
        best_metrics = self._compute_metrics(best_factor)
        
        return {
            'expression': best_expr,
            'factor_values': best_factor,
            'metrics': best_metrics,
            'fitness': best_expr.fitness.values[0],
            'log': log
        }
        
    def plot_evolution(self, log: Dict):
        """绘制进化过程"""
        gen = range(len(log))
        fit_mins = [d['min'] for d in log]
        fit_avgs = [d['avg'] for d in log]
        fit_maxs = [d['max'] for d in log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_mins, 'b-', label='Minimum Fitness', alpha=0.5)
        plt.plot(gen, fit_avgs, 'r-', label='Average Fitness', alpha=0.5)
        plt.plot(gen, fit_maxs, 'g-', label='Maximum Fitness', alpha=0.5)
        
        plt.fill_between(gen, fit_mins, fit_maxs, alpha=0.1)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show() 