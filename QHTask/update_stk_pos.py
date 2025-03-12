
# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


import configparser
from QHConfig.yaml_config import*
import logging
from QHUtil.mongostorager import MongoStorager
from QHUtil.loggers import watch_dog
from QHData.feed import StkDataLoader
from QHFactor.PositionGenerator import StkPosGenerater
from sklearn.pipeline import Pipeline
from QHFactor.SignalGenerator import (
    StkTpotSign,
    StkLGBSign
)
import json
import warnings



warnings.filterwarnings('ignore')
logger = watch_dog(filename='storager.log', level=logging.DEBUG)
logger.info(' > 股票数据及计算目标持仓模块启动...')

# ----------------------------------
# load configuration
cfg = configparser.ConfigParser()
cfg.read(os.path.join(conf_path, 'config.ini'))
stk_conf_path = os.path.join(conf_path, 'stk_conf.yaml')
yd = read_yml(stk_conf_path)

db_settings = cfg['db_settings']
tables = cfg['TABLES']

host = db_settings['DB_NAS_HOST']
port = db_settings['DB_PORT']
db_name = db_settings['DB_NAME']
# ----------------------------------

# %%


def update_skt_lgb5d_pos():
    """
    """
    stk_daily = StkDataLoader(codes=yd['stk_lgbm5d']['codes'], days=252)
    table_name = tables['stk_lgb5d_pos']

    p = Pipeline([
        ('sign', StkLGBSign(days=5)),
        ('pos', StkPosGenerater(weight='rp', periods='W')),
        ('save', MongoStorager(
            host=host,
            table_name=table_name,
         filters=['trade_date', 'code']))
    ])

    p.transform(stk_daily)

    logger.info(
        f'股票机器学习LightGBM5D 策略目标持仓数据已保存至 {host} > {db_name} > {table_name}')


def update_skt_lgb20d_pos():
    """
    """
    stk_daily = StkDataLoader(codes=yd['stk_lgbm5d']['codes'], days=252)
    table_name = tables['stk_lgb20d_pos']

    p = Pipeline([
        ('sign', StkLGBSign(days=20)),
        ('pos', StkPosGenerater(weight='rp', periods='M')),
        ('save', MongoStorager(
            host=host,
            table_name=table_name,
         filters=['trade_date', 'code']))
    ])

    p.transform(stk_daily)

    logger.info(
        f'股票机器学习LightGBM20D 策略目标持仓数据已保存至 {host} > {db_name} > {table_name}')


def update_skt_tpot_5d_pos():
    """
    """
    table_name = tab_name.STK_TPOT_5D_POS.value    # 表名

    p = Pipeline([
        ('sign', StkTpotSign(days=5)),
        ('pos', StkPosGenerater(weight='rp', periods='W')),
        ('save', MongoStorager(table_name=table_name,
         filters=['trade_date', 'code']))
    ])

    p.transform(stk_daily)

    logger.info(f'股票机器学习策略目标持仓数据已保存至 {host} > {db_name} > {table_name}')


def update_skt_tpot_20d_pos():
    """
    """
    table_name = tab_name.STK_TPOT_20D_POS.value    # 表名

    p = Pipeline([
        ('sign', StkTpotSign(days=20)),
        ('pos', StkPosGenerater(weight='rp', periods='M')),
        ('save', MongoStorager(table_name=table_name,
         filters=['trade_date', 'code']))
    ])

    p.transform(stk_daily)

    logger.info(f'股票机器学习策略目标持仓数据已保存至 {host} > {db_name} > {table_name}')


# %%
