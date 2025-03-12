
# %%

import shutil
import zipfile
import pandas as pd
from typing import List, Dict
import requests
from lxml import etree
import re
import json
from QHData.constants import *
import os

"""
期货交易所持仓数据
"""

# 上海金融期货交易所


def get_cffex_holding(trade_date: str) -> List:
    """
    ：上海金融期货交易所会员持仓
    """

    header = {
        'Host': 'www.cffex.com.cn',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.79'
    }

    products = ['IF', 'IC', 'IH', 'IM', 'T', 'TS', 'TF', 'TL']     # 产品列表
    start_urls = [
        f'http://www.cffex.com.cn/sj/ccpm/{trade_date[:6]}/{trade_date[6:]}/{x}.xml' for x in products]

    for url in start_urls:

        r = requests.get(url=url, headers=header,timeout=30,stream=True)

        if r.status_code != 200:
            return None

        select = etree.HTML(r.content)

        # if not select:
        #     raise ValueError('HTML is empty')

        v = select.xpath('//*[@value=0]')
        lng = select.xpath('//*[@value=1]')
        sht = select.xpath('//*[@value=2]')
        try:
            v_dict = [
                {
                    'code': x.xpath('./instrumentid/text()')[0],
                    # 'trade_date':x.xpath('./tradingday/text()')[0],
                    'broker':x.xpath('./shortname/text()')[0],
                    'vol':x.xpath('./volume/text()')[0],
                    'vol_chg':x.xpath('./varvolume/text()')[0]
                }
                for x in v
            ]

            long_dict = [
                {
                    'code': x.xpath('./instrumentid/text()')[0],
                    # 'trade_date':x.xpath('./tradingday/text()')[0],
                    'broker':x.xpath('./shortname/text()')[0],
                    'long_hld':x.xpath('./volume/text()')[0],
                    'long_chg':x.xpath('./varvolume/text()')[0]
                }
                for x in lng
            ]

            short_dict = [
            {
                'code': x.xpath('./instrumentid/text()')[0],
                # 'trade_date':x.xpath('./tradingday/text()')[0],
                'broker':x.xpath('./shortname/text()')[0],
                'short_hld':x.xpath('./volume/text()')[0],
                'short_chg':x.xpath('./varvolume/text()')[0]
            }
            for x in sht
        ]
        except IndexError as e:
            print(e)
            continue

        df = pd.merge(pd.DataFrame(v_dict),
                      pd.merge(pd.DataFrame(long_dict), pd.DataFrame(
                          short_dict), on=['code', 'broker'], how='outer'),
                      on=['code', 'broker'], how='outer').fillna(value=0)

        df['trade_date'] = trade_date
        df = df.replace('-',0)

        dt = df.to_dict(orient='records')
        for d in dt:
            yield d


# 上海期货交易所持仓数据
def get_shfe_holding(trade_date: str) -> List:
    '''
    获取上海期货交易所会员持仓数据
    '''
    url = url_shfe_rank.format(trade_date)
    r = requests.get(url, headers=header, timeout=15)

    if r.status_code != 200:
        return None

    data = json.loads(r.text)['o_cursor']

    if not data:
        return None

    dict_map = {'INSTRUMENTID': 'code',
                'PARTICIPANTABBR1': 'broker',
                'CJ1': 'vol',
                'CJ1_CHG': 'vol_chg',
                'PARTICIPANTABBR2': 'broker',
                'CJ2': 'long_hld',
                'CJ2_CHG': 'long_chg',
                'PARTICIPANTABBR3': 'broker',
                'CJ3': 'short_hld',
                'CJ3_CHG': 'short_chg'
                }

    df = pd.DataFrame(data)[list(dict_map.keys())]
    df.rename(columns=dict_map, inplace=True)
    df = df.applymap(lambda x: str(x).strip())

    df_concat = pd.merge(pd.merge(df.iloc[:, 0:4], df.iloc[:, [0, 4, 5, 6]], on=[
                         'code', 'broker'], how='outer'), df.iloc[:, [0, 7, 8, 9]], on=['code', 'broker'], how='outer').fillna(value=0)

    df_concat = df_concat[df_concat.broker.apply(lambda x:len(x) > 0)]
    df_concat['trade_date'] = trade_date

    grouped = df_concat.groupby('code')
    for _, group in grouped:
        dt = group.to_dict(orient='records')
        for d in dt:
            yield d


# 大连期货交易所持仓数据
def get_dce_holding(trade_date: str) -> List:
    """
    ：大连期货交易所会员持仓数据
    """
    parameters = {
        # memberDealPosiQuotes.variety: a
        'memberDealPosiQuotes.trade_type': '0',
        'contract.contract_id': 'all',
        # 'contract.variety_id': 'all',
        'year': trade_date[:4],
        'month': str(int(trade_date[4:6]) - 1),
        'day': trade_date[-2:],
        'batchExportFlag': 'batch'
    }

    r = requests.post(url=url_dce_rank, data=parameters)
    if r.status_code == 200 and len(r.content) > 800:

        # global file_name
        # global zip_path
        file_name = ''.join([trade_date, '_DCE_DPL.zip'])        # 文件名称
        file_path = '\\'.join([os.getcwd(), file_name])          # 文件压缩路径

        # ------------------------------------------------
        # 下载全市场数据写入压缩文件
        with open(file_path, 'wb') as f:
            f.write(r.content)

        # 解压数据包
        zip_path = file_path.split('.')[0]      # 解压路径
        z = zipfile.ZipFile(file_path, 'r')

        z = z.extractall(zip_path)      # 解压文件
        # z.close()                       # 关闭压缩文件

        if os.path.exists(file_path):
            os.remove(file_path)  # 删除压缩文件

        # ------------------------------------------------

        # 读取每一个合约文件的数据信息
        if os.path.exists(zip_path) == False:
            return

        contracts_paths = ['//'.join([zip_path, x])
                           for x in os.listdir(zip_path)]

        for cp in contracts_paths:
            try:

                with open(cp, 'r', encoding='utf-8') as f:
                    data = f.readlines()
                    data = [re.sub(r'\t+|\s+', ',', i.replace(',', '').strip())
                            for i in data]

            except UnicodeDecodeError:
                with open(cp, 'r', encoding='gbk') as f:
                    data = f.readlines()
                    data = [re.sub(r'\t+|\s+', ',', i.replace(',', '').strip())
                            for i in data]

            try:
                data = [i for i in data if i][1:]
                # if len(data) < 40:
                #     return

                code, date = [x.split('：')[1]
                              for x in data[0].split(',')]    # 获取合约名称和日期

            # code,date = re.findall(r'：(.*?) ',data[0])[0],re.findall(r'\d{4}\-\d{2}\-\d{2}',data[0])[0]

                df = pd.DataFrame(data)
                df = df[df.iloc[:, 0].str.contains('合约|会员|名次|总计') == False]
                df.reset_index(drop=True, inplace=True)
                id = df[df.loc[:, 0].apply(lambda x:x.split(',')[
                    0]) == '1'].index.tolist()

                arr = np.array(df.iloc[:, 0].str.split(',').tolist())

                # 合并信息
                df = pd.merge(
                    pd.merge(pd.DataFrame({'code': code, 'broker': arr[id[0]:id[1], 1], 'vol': arr[id[0]:id[1], 2], 'vol_chg': arr[id[0]:id[1], 3]}),
                             pd.DataFrame({'code': code, 'broker': arr[id[1]:id[2], 1], 'long_hld': arr[id[1]:id[2], 2], 'long_chg': arr[id[1]:id[2], 3]}), on=['code', 'broker'], how='outer'
                             ),
                    pd.DataFrame({'code': code, 'broker': arr[id[2]:, 1], 'short_hld': arr[id[2]:, 2], 'short_chg': arr[id[2]:, 3]}), on=['code', 'broker'], how='outer'
                ).fillna(value=0)

            except IndexError:
                continue

            df['trade_date'] = trade_date
            dt = df.to_dict(orient='records')
            for d in dt:
                yield d

        # if os.path.exists(zip_path):
        shutil.rmtree(zip_path)     # 递归删除解压文件夹 删除空文件夹


# 郑州期货交易所持仓数据
def get_czce_holding(trade_date: str) -> List:
    """
    ：郑州交易所持仓数据
    """
    url = url_czce_rank.format(trade_date[:4], trade_date)
    r = requests.get(url, headers=header, timeout=15)
    r.encoding = 'utf-8'

    if r.status_code != 200:
        return

    # 商品列表
    df_html = pd.read_html(r.text)

    df = df_html[0] if len(df_html[0]) > 20 else df_html[1]
    # codes = df[df.iloc[:, 0].str.contains('品种|合约')][0].apply(
    #     lambda x: x.split('  ')[0].split('：')[1]).tolist()

    codes = df[df.iloc[:, 0].str.contains('品种|合约')][0].apply(
        lambda x: re.findall(r'：(.*?) ', x)[0]).tolist()
    df.loc[(df[0] == '1'), 'code'] = codes

    df = df[df.iloc[:, 0].str.contains('品种|合约|名次|合计') == False]
    df.fillna(method='ffill', inplace=True)          # 填充缺失值

    result = pd.merge(pd.merge(df.loc[:, ['code', 1, 2, 3]].rename(columns={1: 'broker', 2: 'vol', 3: 'vol_chg'}),
                      df.loc[:, ['code', 4, 5, 6]].rename(columns={4: 'broker', 5: 'long_hld', 6: 'long_chg'}), on=['code', 'broker'], how='outer'
    ), df.loc[:, ['code', 7, 8, 9]].rename(columns={7: 'broker', 8: 'short_hld', 9: 'short_chg'}), on=['code', 'broker'], how='outer').fillna(value=0)

    # # drop broker equal '-'
    result = result[result['broker'].apply(lambda x:len(x) > 1)]

    result['trade_date'] = trade_date
    result = result.applymap(lambda x: str(x).replace('-', 0))

    grouped = result.groupby('code')
    for _, group in grouped:
        dt = group.to_dict(orient='records')
        for d in dt:
            yield d


# 获取全部期货交易所持仓数据
def get_fut_holding(trade_date: str):
    """
    ：获取期货交易所会员持仓数据
    ：trade_date        交易日期
    """

    exchangs = ['SHFE', 'DCE', 'CZCE', 'CFFEX']
    for x in exchangs:
        try:
            if x == 'CFFEX':
                cfx_data = get_cffex_holding(trade_date=trade_date)
                for d in cfx_data:
                    yield d

            if x == 'SHFE':
                shfe_data = get_shfe_holding(trade_date=trade_date)
                for d in shfe_data:
                    yield d

            if x == 'DCE':
                dce_data = get_dce_holding(trade_date=trade_date)
                for d in dce_data:
                    yield d

            if x == 'CZCE':
                czce_data = get_czce_holding(trade_date=trade_date)
                
                for d in czce_data:
                    yield d

        except (KeyError, TypeError):
            continue
