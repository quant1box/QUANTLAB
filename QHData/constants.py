import numpy as np

# 新浪期货成交持仓地址
fut_symbols = "http://vip.stock.finance.sina.com.cn/q/view/vFutures_Positions_cjcc.php"
fut_pos = 'http://vip.stock.finance.sina.com.cn/q/view/vFutures_Positions_cjcc.php?t_breed={0}&t_date={1}'

# 交易所行情地址

url_cffex_daily = 'http://www.cffex.com.cn/sj/hqsj/rtj/{}/{}/index.xml'
url_shfe_daily = 'http://www.shfe.com.cn/data/dailydata/kx/kx{0}.dat'

url_czce_daily = 'http://www.czce.com.cn/cn/DFSStaticFiles/Future/{0}/{1}/FutureDataDaily.htm'
url_dce_daily = 'http://www.dce.com.cn/publicweb/quotesdata/dayQuotesCh.html'

# 交易所成交排名地址
url_cffex_rank = 'http://www.cffex.com.cn/sj/ccpm/{}/{}.xml'  # 日期 xxxx/xx,品种
# url_cffex_rank = 'http://www.cffex.com.cn/sj/ccpm/{}/T.xml'
url_shfe_rank = 'http://www.shfe.com.cn/data/dailydata/kx/pm{}.dat'
url_czce_rank = 'http://www.czce.com.cn/cn/DFSStaticFiles/Future/{0}/{1}/FutureDataHolding.htm'
url_dce_rank = 'http://www.dce.com.cn/publicweb/quotesdata/exportMemberDealPosiQuotesBatchData.html'


# 生意社基差等地址
url_basis = 'http://www.100ppi.com/sf2/day-{}.html'
url_rollover = 'http://www.100ppi.com/sf/day-{}.html'

# 上海黄金交易所每日行情
url_sge = 'https://www.sge.com.cn/sjzx/mrhqsj'

# 交易所商品代码
map_market = {'CFFEX': {'沪深300股指期货': 'IF', '中证500股指期货': 'IC', '上证50股指期货': 'IH', '10年期国债期货': 'T',
                        '5年期国债期货': 'TF', '2年期国债期货': 'TS'},

              'DCE': {'豆一': 'A', '豆二': 'B', '豆粕': 'M', '豆油': 'Y', '棕榈油': 'P', '玉米': 'C', '玉米淀粉': 'CS', '鸡蛋': 'JD',
                      '纤维板': 'FB', '胶合板': 'BB', '聚乙烯': 'L', '聚氯乙烯': 'V', '聚丙烯': 'PP', '焦炭': 'J', '焦煤': 'JM',
                      '铁矿石': 'I', '乙二醇': 'EG', '生猪': 'LH', '苯乙烯': 'EB', '液化石油气': 'PG'
                      },

              'CZCE': {'强麦': 'WH', '普麦': 'PM', '棉花': 'CF', '白糖': 'SR', 'PTA': 'TA', '菜籽油': 'OI', '早籼': 'RI',
                       '甲醇': 'MA', '玻璃': 'FG', '油菜籽': 'RS', '菜籽粕': 'RM', '动力煤': 'ZC', '粳稻': 'JR', '晚籼': 'LR',
                       '硅铁': 'SF', '锰硅': 'SM', '棉纱': 'CY', '苹果': 'AP', '尿素': 'UR', '纯碱': 'SA', '涤纶短纤': 'PF'},

              'SHFE': {'铜': 'CU', '铝': 'AL', '锌': 'ZN', '铅': 'PB', '镍': 'NI', '锡': 'SN', '黄金': 'AU', '白银': 'AG',
                       '螺纹钢': 'RB', '线材': 'WR', '热轧卷板': 'HC', '燃料油': 'FU', '石油沥青': 'BU', '天然橡胶': 'RU', '原油': 'SC',
                       '纸浆': 'SP', '不锈钢': 'SS'},

              'INE': {'原油': 'SC'}
              }


instruments = np.concatenate([[x for x in v.values()]
                             for k, v in map_market.items()]).tolist()

header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
}


# 南华期货指数
# fut_index = {'NHAI.NH': '南华农产品指数',
#              'NHCI.NH': '南华商品指数',
#              'NHECI.NH': '南华能化指数',
#              'NHFI.NH': '南华黑色指数',
#              'NHII.NH': '南华工业品指数',
#              'NHMI.NH': '南华金属指数',
#              'NHNFI.NH': '南华有色金属',
#              'NHPMI.NH': '南华贵金属指数',
#              'A.NH': '南华连大豆指数',
#              'AG.NH': '南华沪银指数',
#              'AL.NH': '南华沪铝指数',
#              'AP.NH': '南华郑苹果指数',
#              'AU.NH': '南华沪黄金指数',
#              'BB.NH': '南华连胶合板指数',
#              'BU.NH': '南华沪石油沥青指数',
#              'C.NH': '南华连玉米指数',
#              'CF.NH': '南华郑棉花指数',
#              'CS.NH': '南华连玉米淀粉指数',
#              'CU.NH': '南华沪铜指数',
#              'CY.NH': '南华棉纱指数',
#              'ER.NH': '南华郑籼稻指数',
#              'FB.NH': '南华连纤维板指数',
#              'FG.NH': '南华郑玻璃指数',
#              'FU.NH': '	南华沪燃油指数',
#              'HC.NH': '南华沪热轧卷板指数',
#              'I.NH': '南华连铁矿石指数',
#              'J.NH': '南华连焦炭指数',
#              'JD.NH': '南华连鸡蛋指数',
#              'JM.NH': '南华连焦煤指数',
#              'JR.NH': '南华郑粳稻指数',
#              'L.NH': '南华连乙烯指数',
#              'LR.NH': '南华郑晚籼稻指数',
#              'M.NH': '南华连豆粕指数',
#              'ME.NH': '南华郑甲醇指数',
#              'NI.NH': '南华沪镍指数',
#              'P.NH': '南华连棕油指数',
#              'PB.NH': '南华沪铅指数',
#              'PP.NH': '南华连聚丙烯指数',
#              'RB.NH': '南华沪螺钢指数',
#              'RM.NH': '南华郑菜籽粕指数',
#              'RO.NH': '南华郑菜油指数',
#              'RS.NH': '南华郑油菜籽指数',
#              'RU.NH': '南华沪天胶指数',
#              'SC.NH': '南华原油指数',
#              'SF.NH': '南华郑硅铁指数',
#              'SM.NH': '南华郑锰硅指数',
#              'SN.NH': '南华沪锡指数',
#              'SP.NH': '南华纸浆指数',
#              'SR.NH': '南华郑白糖指数',
#              'TA.NH': '南华郑精对苯二甲酸指数',
#              'TC.NH': '南华郑动力煤指数',
#              'V.NH': '南华连聚氯乙烯指数',
#              'WR.NH': '南华沪线材指数',
#              'WS.NH': '南华郑强麦指数',
#              'Y.NH': '南华连豆油指数',
#              'ZN.NH': '南华沪锌指数'}
