#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from QHData.data_base import (
    fut_holding,
    fut_spot,
    basis
)

from QHData.fut_holding import get_fut_holding
from peewee import IntegrityError
from QHData.ppi import ppi


def update_basis_data(trade_date: str):
    """更新商品期货主力基差"""

    p = ppi()
    data = p.basis(trade_date=trade_date)

    for d in data:
        try:

            basis.insert(d).execute()

        except IntegrityError as e:
            print(e)


def update_fut_spot_data(trade_date: str):
    """更新现货期货基差对比表"""

    p = ppi()
    data = p.fut_spot(trade_date=trade_date)

    for d in data:
        try:

            fut_spot.insert(d).execute()

        except IntegrityError as e:
            print(e)


def update_fut_holding_data(trade_date: str):
    """更新现货期货基差对比表"""

    data = get_fut_holding(trade_date=trade_date)

    for d in data:
        try:

            fut_holding.insert(d).execute()

        except IntegrityError as e:
            print(e)


# if __name__ == '__main__':

#     cur_date = '20231017'

#     # update_basis_data(trade_date=cur_date)
#     # update_fut_spot_data(trade_date=cur_date)
#     update_fut_holding_data(trade_date=cur_date)

# %%
