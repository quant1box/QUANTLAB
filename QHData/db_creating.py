

from peewee import (
    CompositeKey,
    PostgresqlDatabase,
    Model,
    DateField,
    CharField,
    IntegerField,
    FloatField
)


db = PostgresqlDatabase('quantaxis', user='chiang',
                        password='035115', host='192.168.31.220', port=5433)

# 期货日线数据
class fut_daily(Model):

    trade_date = DateField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()

    settle = FloatField()
    pre_settle = FloatField()
    ch1 = FloatField()
    ch2 = FloatField()
    vol = IntegerField()
    vol_chg = IntegerField()

    class Meta:
        database = db
        primary_key = CompositeKey('trade_date', 'code')

# 期权日线数据


# 会员持仓排名
class fut_holding(Model):
    trade_date = DateField()
    code = CharField()
    broker = CharField()
    vol = IntegerField()
    vol_chg = IntegerField()
    long_hld = IntegerField()
    long_chg = IntegerField()
    short_hld = IntegerField()
    short_chg = IntegerField()

    class Meta:
        database = db
        # indexes = (
        #     (('trade_date', 'symbol', 'broker'), True),
        # )
        primary_key = CompositeKey('trade_date', 'code', 'broker')


# 生意社基差表
class basis(Model):

    trade_date = DateField()
    code = CharField()
    close = FloatField()
    spot_price = FloatField()

    basis = FloatField()
    basis_ratio = FloatField()
    basis_max_180 = FloatField()
    basis_min_180 = FloatField()
    basis_avg_180 = FloatField()

    class Meta:
        database = db
        # indexs = (
        #     (('trade_date', 'code'), True),
        # )
        primary_key = CompositeKey('trade_date', 'code')


# 生意社现货期货价格对比表
class fut_spot(Model):

    trade_date = DateField()
    spot_price = FloatField()
    latest_code = CharField()
    latest_price = FloatField()
    latest_basis = FloatField()
    latest_basis_ratio = FloatField()

    code = CharField()
    close = FloatField()
    basis = FloatField()
    basis = FloatField()
    basis_ratio = FloatField()

    class Meta:
        database = db
        # indexs = (
        #     (('trade_date', 'code'), True)
        # )
        primary_key = CompositeKey('trade_date', 'code')


# Baostock 股票分钟线数据
class stk_min(Model):
    date = DateField()
    time = DateField()
    code = CharField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()
    amount = FloatField()
    adjustflag = FloatField()

    class Meta:
        database = db
        primary_key = CompositeKey('date', 'time', 'code')


# TODO  检查字段
# Baostock股票周
class stock(Model):

    date = DateField()
    code = CharField()
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    preclose = FloatField()
    volume = FloatField()
    amount = FloatField()
    adjustflag = FloatField()
    turn = FloatField()
    tradestatus = FloatField()
    pctChg = FloatField()
    isST = FloatField()

    class Meta:
        database = db
        primary_key = CompositeKey('date', 'code')


if __name__ == '__main__':

    db.connect()
    db.create_tables([fut_holding, basis, fut_spot,stock,stk_min])
