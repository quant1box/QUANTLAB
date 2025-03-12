# %%
import re
from time import sleep
from typing import Dict, List
from numpy import product
import pandas as pd
from pandas.io import json
import requests
from lxml import etree
from datetime import datetime, time

header = {
    'Host': 'analyse.kiiik.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36'
}


class KiiikSpider:
    def __init__(self):
        self.exchanges = ['CFFEX', 'SHFE', 'DCE', 'CZCE', 'INE']
        self.start_url = "http://vip.stock.finance.sina.com.cn/q/view/vFutures_Positions_cjcc.php"

    def get_symbols(self) -> List:
        r = requests.get(self.start_url, timeout=5)
        r.encoding = 'gbk'
        select = etree.HTML(r.text)
        symbols_list = select.xpath('//select/option/@value')
        return symbols_list

    def get_exchange_hold_positions(self, trade_date: str):
        for exchg in self.exchanges:
            url_products = 'https://analyse.cesfutures.com/phr/getExchangeHoldProducts/{}/{}.html'.format(
                exchg, trade_date)
            r1 = requests.get(url=url_products, headers=header, timeout=30)
            d1 = json.loads(r1.text)
            if not d1:
                break
            products_list = [x.get('value') for x in d1[0]]
            for p in products_list:
                url_contracts = 'https://analyse.cesfutures.com/phr/getExchangeHoldContracts/{}/{}.html'.format(
                    p, trade_date)
                r2 = requests.get(url=url_contracts,
                                  headers=header, timeout=30)
                d2 = json.loads(r2.text)
                if not d2:
                    break
                if isinstance(d2, list):
                    for c in d2:
                        url_positions = 'https://analyse.cesfutures.com/phr/getExchangeHoldPositions/{}/{}/{}.html'.format(
                            exchg, c, trade_date)
                        r3 = requests.get(url=url_positions, headers=header)
                        d3 = json.loads(r3.text)
                        df = pd.DataFrame(d3)
                        result = pd.merge(
                            pd.merge(df.loc[:, ['secuid', 'company1', 'transAmt1', 'changeAmt1']].rename(columns={'secuid': 'symbol', 'company1': 'broker', 'transAmt1': 'vol', 'changeAmt1': 'vol_chg'}),
                                     df.loc[:, ['secuid', 'company2', 'transAmt2', 'changeAmt2']].rename(
                                         columns={'secuid': 'symbol', 'company2': 'broker', 'transAmt2': 'long_hld', 'changeAmt2': 'long_chg'}),
                                     on=['symbol', 'broker'], how='outer'),
                            df.loc[:, ['secuid', 'company3', 'transAmt3', 'changeAmt3']].rename(
                                columns={'secuid': 'symbol', 'company3': 'broker', 'transAmt3': 'short_hld', 'changeAmt3': 'short_chg'}),
                            on=['symbol', 'broker'], how='outer'
                        ).fillna(value=0)
                        result['trade_date'] = trade_date
                        result = result[result.broker.apply(
                            lambda x:len(x) > 1)]
                        result = result[result.symbol.apply(
                            lambda x:len(x) > 0)]
                        dt = result.to_dict(orient='records')
                        for d in dt:
                            yield d
                            sleep(.05)

    def get_long_short_in_or_decrease_data(self, trade_date: str) -> pd.DataFrame:
        url = 'http://analyse.kiiik.com/phr/getLongShortInOrDecreaseData/{}.html'.format(
            trade_date)
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            if r.text:
                data = json.loads(r.text)
            if not data:
                return
            df = pd.DataFrame(data)
            df.drop(columns=['id', 'buySellRate2'], inplace=True)
            df.rename(columns={'recordDate': 'trade_date',
                      'secuid': 'symbol'}, inplace=True)
            return df

    def get_long_short_in_quote_data(self, trade_date: str) -> pd.DataFrame:
        url = 'http://analyse.kiiik.com/phr/getLongShortInQuoteDate/{}.html'.format(
            trade_date)
        r = requests.get(url, headers=header)
        if r.status_code == 200:
            data = json.loads(r.text)
            if not data:
                return None
            df = pd.DataFrame(data)
            df[['exchangeId', 'secuid']] = df[[
                'exchangeId', 'secuid']].applymap(lambda x: x.strip())
            return df
        else:
            return None

    def get_long_short_rate_data(self, trade_date: str) -> pd.DataFrame:
        url = 'http://analyse.kiiik.com/phr/getLongShortRateData/{}.html'.format(
            trade_date)
        r = requests.get(url, headers=header)
        if r.status_code == 200:
            data = json.loads(r.text)
            if not data:
                return None
            df = pd.DataFrame(
                {'secuid': data[0], 'rate': data[1], 'trade_date': trade_date})
            return df


# if __name__ == "__main__":

#     kiiik = KiiikSpider()
#     data = kiiik.get_exchange_hold_positions(trade_date='20230601')
#     for d in data:
#         print(d)

# %%
