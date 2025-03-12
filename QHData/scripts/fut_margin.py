# 获取期货保证金比率

# %%
import json
import re
from typing import Dict, List
import pandas as pd
import requests
from lxml import etree
from datetime import datetime, time
from time import sleep


def extractPID(rawCode: str):
    idx = 0
    for c in rawCode:
        if c.isalpha():
            idx += 1
        else:
            break
    return rawCode[:idx]


def get_margin_list():
    url = 'https://www.9qihuo.com/qihuoshouxufei'

    header = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }

    r = requests.get(url=url, headers=header, verify=False)
    select = etree.HTML(r.text)
    tr_list = select.xpath('//tr')

    margin_list = []
    for td in tr_list[3:]:
        try:

            pattern = r'\d+\.\d+|\d+|\-\d+\.\d+|\-\d+'

            code = td.xpath('./td/a/b/text()')[0]
            # price = td.xpath('./td[2]/text()')[0]
            # 'buy_margin': td.xpath('//*[@id="heyuetbl"]/tbody/tr[4]/td[2]')[0],
            # 'sell_margin': td.xpath('//*[@id="heyuetbl"]/tbody/tr[4]/td[5]')[0],
            cash = td.xpath('./td[6]/text()')[0]

            pid = extractPID(code)
            month = code[len(pid):]
            if len(month) == 3:
                month = "2" + month

            code = ''.join([pid, month])

            margin = re.findall(pattern, cash)[0]

            print({'合约品种': code,
                   '保证金/每手': float(margin)
                   })

            margin_list.append({'code': code, 'margin': float(margin)})

        except:
            continue

        sleep(.1)

    return margin_list


if __name__ == '__main__':
    margin_list = get_margin_list()
    with open('fut_margin.json', 'w') as f:
        json.dump(margin_list, f)
