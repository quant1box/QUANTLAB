import cvxpy as cp
import numpy as np


class PortfolioOptimizer:
    """
    投资组合优化器，支持多种优化目标（如最小化风险、最大化夏普比率等）。
    """

    def __init__(self,
                 returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 industry_labels: np.ndarray = None,
                 lower_bound: float = 0.01,
                 upper_bound: float = 0.02,
                 industry_weight_limit: float = 0.065):
        """
        初始化投资组合优化器。

        :param returns: 资产的收益率矩阵，形状为 (样本数, 资产数)
        :param cov_matrix: 资产的协方差矩阵，形状为 (资产数, 资产数)
        :param industry_labels: 资产的行业标签，形状为 (资产数,)，可选
        :param lower_bound: 权重的下界，默认为0.01
        :param upper_bound: 权重的上界，默认为0.02
        :param industry_weight_limit: 行业权重的上限，默认为0.065
        """
        if not isinstance(returns, np.ndarray) or not isinstance(cov_matrix, np.ndarray):
            raise ValueError("returns和cov_matrix必须是numpy数组。")
        if returns.ndim != 2 or cov_matrix.ndim != 2:
            raise ValueError("returns和cov_matrix必须是二维数组。")
        if returns.shape[1] != cov_matrix.shape[0]:
            raise ValueError("returns的列数必须等于cov_matrix的行数。")

        self.returns = returns
        self.cov_matrix = cov_matrix
        self.industry_labels = industry_labels
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.industry_weight_limit = industry_weight_limit
        self.num_assets = cov_matrix.shape[0]
        self.weights = cp.Variable(self.num_assets)  # 权重变量

    def _check_problem_status(self, problem):
        """
        检查优化问题的求解状态。

        :param problem: cvxpy问题对象
        :raises ValueError: 如果问题未成功解决，则抛出异常
        """
        if problem.status != cp.OPTIMAL:
            raise ValueError(f"求解失败，状态: {problem.status}")

    def _get_constraints(self):
        """
        获取权重约束条件。

        :return: 约束条件列表
        """
        constraints = [
            cp.sum(self.weights) == 1,         # 权重和必须为1
            self.weights >= self.lower_bound,  # 权重的下界
            self.weights <= self.upper_bound   # 权重的上界
        ]

        # 添加行业约束（如果提供了行业标签）
        if self.industry_labels is not None:
            for industry in np.unique(self.industry_labels):
                industry_mask = (self.industry_labels == industry)
                constraints.append(
                    cp.sum(self.weights[industry_mask]) <= self.industry_weight_limit)

        return constraints

    def minimize_risk(self):
        """
        最小化组合风险（方差）。

        :return: 优化后的权重
        """
        # 定义目标函数：最小化组合方差
        portfolio_variance = cp.quad_form(self.weights, self.cov_matrix)
        objective = cp.Minimize(portfolio_variance)

        # 定义优化问题并求解
        constraints = self._get_constraints()
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self._check_problem_status(problem)

        return self.weights.value
    
    
    def maximize_sharpe(self, risk_free_rate=0.0):
        """
        最大化夏普比率，调整为凸优化形式 (SOCP)
        
        :param risk_free_rate: 无风险利率，默认为0.0
        :return: 优化后的权重
        """
        # 期望收益
        expected_return = cp.sum(cp.multiply(self.returns.mean(axis=0), self.weights))

        # 投资组合风险
        portfolio_risk = cp.quad_form(self.weights, self.cov_matrix)

        # 目标：最大化夏普比率
        objective = cp.Maximize((expected_return - risk_free_rate) / (cp.sqrt(portfolio_risk) + 1e-10))  # 添加小常数以避免除以零

        # 约束条件
        constraints = [
            self.weights >= self.lower_bound,  # 权重下界
            self.weights <= self.upper_bound,  # 权重上界
            cp.sum(self.weights) == 1,        # 权重和为1
        ]

        # 定义并求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=True)

        # 检查问题状态
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        # 返回最优权重
        return self.weights.value

    def maximize_return(self):
        """
        最大化组合收益。

        :return: 优化后的权重
        """
        # 目标函数：最大化组合收益
        portfolio_return = cp.sum(cp.multiply(self.returns.mean(axis=0), self.weights))
        objective = cp.Maximize(portfolio_return)

        # 定义优化问题并求解
        constraints = self._get_constraints()
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self._check_problem_status(problem)

        return self.weights.value

    def maximize_information_ratio(self, benchmark_returns: np.ndarray):
        """
        最大化信息比率。

        :param benchmark_returns: 基准收益率，形状为 (样本数,)
        :return: 优化后的权重
        """
        if not isinstance(benchmark_returns, np.ndarray) or benchmark_returns.ndim != 1:
            raise ValueError("benchmark_returns必须是一维numpy数组。")

        # 组合的超额收益
        portfolio_return = cp.sum(cp.multiply(self.returns.mean(axis=0), self.weights))
        excess_return = portfolio_return - benchmark_returns.mean()
        # 跟踪误差
        tracking_error = cp.sqrt(cp.quad_form(self.weights, self.cov_matrix))

        # 添加约束以确保跟踪误差有效
        constraints = self._get_constraints()
        constraints.append(tracking_error >= 1e-10)

        # 信息比率
        information_ratio = excess_return / (tracking_error + 1e-10)
        objective = cp.Maximize(information_ratio)

        # 定义优化问题并求解
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self._check_problem_status(problem)

        return self.weights.value

    def maximize_sortino_ratio(self, target_return: float = 0.0):
        """
        最大化索提诺比率。

        :param target_return: 目标收益率，默认为0.0
        :return: 优化后的权重
        """
        # 组合的超额收益
        portfolio_return = cp.sum(cp.multiply(self.returns.mean(axis=0), self.weights))
        excess_return = portfolio_return - target_return

        # 下行风险（只考虑负收益的平方和）
        downside_risk = cp.sqrt(
            cp.sum(cp.pos(target_return - (self.returns @ self.weights)) ** 2) / self.returns.shape[0]
        )

        # 添加约束以确保下行风险有效
        constraints = self._get_constraints()
        constraints.append(downside_risk >= 1e-10)

        # 索提诺比率
        sortino_ratio = excess_return / (downside_risk + 1e-10)
        objective = cp.Maximize(sortino_ratio)

        # 定义优化问题并求解
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self._check_problem_status(problem)

        return self.weights.value