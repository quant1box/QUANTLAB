
# %%
import sys
sys.path.append('..')

import warnings
from QHFactor.factors_calclatror import AlphaCalculator
from QHMlflow.mlflowtracker import load_model_and_config
from datetime import datetime, timedelta
from QHData.data_load import DataLoader
from sklearn.pipeline import Pipeline
from QHMlflow.generator import (
    SignalsGenerator,
    PositionGenerator,
    MongoPositionStorage
)
from QHUtil.loggers import watch_dog
from QHFactor.transformer import (
    RollingWinsorizeTransformer,
    RollingZScoreTransformer
)

warnings.filterwarnings('ignore')
logger = watch_dog(filename='task.log')



# 生成目标持仓基类
class PosBaseStrategy:

    def __init__(self,
                 track_uri: str,
                 experiment_name: str,
                 run_id: str,
                 datasets: str,
                 ) -> None:

        # 獲取模型和配置文件
        self.model, self.config = load_model_and_config(
            experiment_name=experiment_name,
            remote_tracking_uri=track_uri,
            run_id=run_id
        )

        # 获取当前日期
        current_date = datetime.now()

        # 计算最近一年的开始日期
        start_date = current_date - timedelta(days=100)

        start = start_date.strftime('%Y-%m-%d')
        end = current_date.strftime('%Y-%m-%d')

        # 加載數據
        self.fut_daily = DataLoader(start, end).load_future(
            codes=self.config['codes'])

        processor = Pipeline([
            ('claculate', AlphaCalculator(alphas=self.config['features'])),
            ('roll_winsoriz', RollingWinsorizeTransformer()),
            ('roll_zscore', RollingZScoreTransformer())
        ])

        self.features = processor.fit_transform(self.fut_daily)

        # 保存目标数据表
        self.datasets = datasets

    def generate_positions(self, threshold: float = 0.001):
        """"""
        p = Pipeline([
            ('generate signals', SignalsGenerator(self.model, threshold, True)),    # 生成信号
            ('genetate positions', PositionGenerator(price_data=self.fut_daily)),   # 生成持仓
            ('store2mongo', MongoPositionStorage(collection_name=self.datasets))    # 存数据库
        ])

        pos = p.transform(self.features)
        print(pos)

strategies = [
    {
        'experiment_name': 'FUT_LGBMREG_1D',
        'run_id': '7b91679b9e0646b58f6807f575e57385',
        'table_name': 'FUT_LGBMREG1D_POS',
        'threshold': [-0.005,0.005]
    },
    {
        'experiment_name': 'FUT_LGBMRANK_1D',
        'run_id': 'ccdc3e026e284637abfc42a254370374',
        'table_name': 'FUT_LGBMRANK1D_POS',
        'threshold': [-0.025,0.025]
    },
]


def update_positions():

    track_uri = 'http://192.168.31.220:5115/'

    for strategy in strategies:
        
        exp_name = strategy['experiment_name']
        run_id = strategy['run_id']
        table_name = strategy['table_name']
        threshold = strategy['threshold']

        print(exp_name, run_id, table_name)
        pos = PosBaseStrategy(
            track_uri=track_uri,
            experiment_name=exp_name,
            run_id=run_id,
            datasets=table_name,
        )
        pos.generate_positions(threshold=threshold)

        logger.info(
            f'The target position of strategy {exp_name} has been updated to data table {table_name}')
