# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json
from lxml import etree
from QHData.constants import *
import time
import requests
import re
import pandas as pd
from typing import Dict


# %%
# 获取生意社想期限数据

class ppi:

    def __init__(self) -> None:
        # 合约映射表
        self.map_dict = {
            "铜": "cu",
            "强麦WH":"wh",
            "螺纹钢": "rb",
            "锌": "zn",
            "铝": "al",
            "黄金": "au",
            "线材": "wr",
            "燃料油": "fu",
            "天然橡胶": "ru",
            "铅": "pb",
            "白银": "ag",
            "石油沥青": "bu",
            "热轧卷板": "hc",
            "镍": "ni",
            "锡": "sn",
            "苹果": "AP",
            "纸浆": "sp",
            "不锈钢": "ss",
            "丁二烯橡胶": "br",
            "PTA": "TA",
            "白糖": "SR",
            "棉花": "CF",
            "普麦": "pm",
            "菜籽油OI": "OI",
            "玻璃": "FG",
            "菜籽粕": "RM",
            "油菜籽": "rs",
            "硅铁": "sf",
            "锰硅": "sm",
            "甲醇MA": "MA",
            "动力煤ZC": "ZC",
            "棉纱": "cy",
            "尿素": "ur",
            "纯碱": "SA",
            "涤纶短纤": "pf",
            "PX": "px",
            "烧碱": "sh",
            "棕榈油": "p",
            "聚氯乙烯": "v",
            "聚乙烯": "l",
            "豆一": "a",
            "豆粕": "m",
            "豆油": "y",
            "玉米": "c",
            "焦炭": "j",
            "焦煤": "jm",
            "铁矿石": "i",
            "鸡蛋": "jd",
            "聚丙烯": "pp",
            "玉米淀粉": "cs",
            "乙二醇": "eg",
            "苯乙烯": "eb",
            "液化石油气": "pg",
            "生猪": "lh",
            "工业硅": "si",
            "碳酸锂": "lc"
        }

    def parse_page(self, url):
        """网页解析"""
        try:
            url = url
            response = requests.get(url, timeout=15)
            response.encoding = 'utf-8'
            response.raise_for_status()

        except Exception as e:
            print(f"Request failed: {e}")
            return

        html = etree.HTML(response.text)

        if not html:
            raise ValueError('HTML is empty')

        # if not html.getroot():
            # raise ValueError('Failed to parse HTML')

        return html


    def basis(self, trade_date: str):
        """商品主力基差数据"""

        basis_url = url_basis.format(
            '-'.join([trade_date[:4], trade_date[4:6], trade_date[-2:]]))
        select = self.parse_page(url=basis_url)

        tr = select.xpath('//tr[@align]')

        df = pd.DataFrame({
            'trade_date': trade_date,                                           # 交易日期
            'code':[''.join([self.map_dict.get(x.xpath('./td/a/text()')[0]),x.xpath('./td[3]/text()')[0].strip()]) for x in tr],
            'close': [x.xpath('./td[4]/text()')[0] for x in tr],
            'spot_price': [x.xpath('./td[2]/text()')[0] for x in tr],    
            'basis': [x.xpath('./td[5]//tr/td[1]//text()')[0] for x in tr],
            'basis_ratio': [x.xpath('./td[5]//tr/td[2]//text()')[0].replace('%','') for x in tr],
            'basis_max_180': [x.xpath('./td[6]/text()')[0] for x in tr],
            'basis_min_180': [x.xpath('./td[7]/text()')[0] for x in tr],
            'basis_avg_180': [x.xpath('./td[last()]/text()')[0] for x in tr]
        })

        df = df.apply(lambda x:x.str.strip())
        df = df.replace('',0)
        for row in df.to_dict(orient='records'):
            yield row


    def fut_spot(self,trade_date:str):
        """期货与现货价格对比表"""

        url = url_rollover.format(
        '-'.join([trade_date[:4], trade_date[4:6], trade_date[-2:]]))
        select = self.parse_page(url=url)

        tr = select.xpath('//tr[@align]')

        df = pd.DataFrame({
            'trade_date':trade_date,
            'spot_price':[x.xpath('./td[2]/text()')[0].strip() for x in tr],
            'latest_code':[''.join([self.map_dict.get(x.xpath('./td/a/text()')[0]),x.xpath('./td[3]/text()')[0].strip()]) for x in tr],
            "latest_price":[x.xpath('./td[4]/text()')[0] for x in tr],
            "latest_basis":[x.xpath('./td[5]//tr//td[1]//text()')[0] for x in tr],
            "latest_basis_ratio":[x.xpath('./td[5]//tr//td[2]//text()')[0].replace('%','') for x in tr],
            "code":[''.join([self.map_dict.get(x.xpath('./td/a/text()')[0]),x.xpath('./td[6]/text()')[0].strip()]) for x in tr],
            "close":[x.xpath('./td[7]/text()')[0] for x in tr],
            'basis':[x.xpath('./td[8]//tr//td[1]//text()')[0] for x in tr],
            'basis_ratio':[x.xpath('./td[8]//tr//td[2]//text()')[0].replace('%','') for x in tr]
        })
       
        df = df.apply(lambda x:x.str.strip())
        df = df.replace('',0)
        for row in df.to_dict(orient='records'):
            yield row

