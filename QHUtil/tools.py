
# %%

import shutil
import inspect
import os
import pickle
import json

try:
    import QUANTAXIS as QA
except ImportError as e:
    print("Error importing QUANTAXIS module:", e)

from typing import Dict, List, Union


def load_config_file_path(filename: str):
    """
    获取配置文件路径

    :return: 配置文件路径
    """
    return os.path.join(os.path.dirname(__file__), '..', 'QHConfig', filename)


def read_json(file_path):
    """
    读取 JSON 文件并返回解析后的 Python 对象

    参数：
      file_path: JSON 文件的路径

    返回：
      解析后的 Python 对象
    """
    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None


def extractPID(rawCode: str):
    """
    抽取合约产品code
    """
    idx = 0
    for c in rawCode:
        if c.isalpha():
            idx += 1
        else:
            break
    return rawCode[:idx]


def pInfo(code: str) -> Dict:
    """
    获取商品相关信息
    code: short code
    """

    data = read_json(load_config_file_path('commodities.json'))
    dit = {m.upper(): n for k, v in data.items() for m, n in v.items()}

    return dit.get(code, None)


def hot_map(code: str):
    """
    获取主力合约或者
    : code 合约标识(RB、SR、TA)
    """

    data = data = read_json(load_config_file_path('hotmap.json'))
    dit = {m.upper(): n for k, v in data.items() for m, n in v.items()}

    return dit.get(code, None)


def sec_map(code: str):
    """
    获取次主力合约
    : code 合约标识(RB、RU、BU、SR)
    """

    data = data = read_json(load_config_file_path('secmap.json'))
    dit = {m.upper(): n for k, v in data.items() for m, n in v.items()}

    return dit.get(code, None)


def chunks(data: List, chunk_size: int):
    """
    数据分片
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]


def calc_trade_date(current_date: str):
    """
    计算下一个交易日
    """
    target_date = QA.QA_util_get_pre_trade_date(
        cursor_date=current_date, n=-1).replace('-', '')

    return target_date


def transform_stdCode(code: str) -> str:
    """
    : 转换成wtpy平台需要的标准合约
    : code SR302,rb2301 .etc
    : return eg. SHFE.rb.2301
    """

    pid = extractPID(code)

    exchg = pInfo(pid.upper())['exchg']
    month = code[len(pid):]
    if len(month) == 3:
        month = ''.join(['2', month])

    stdCode = '.'.join([exchg, pid, month])

    return stdCode


def transform_hot(code: str) -> str:
    """
    : 转换成wtpy平台需要的标准合约
    : code SR302,rb2301 .etc
    : return eg. SHFE.rb.HOT
    """
    pid = extractPID(code)

    exchg = pInfo(pid.upper())['exchg']
    month = code[len(pid):]
    if len(month) == 3:
        month = ''.join(['2', month])
    code = '.'.join([exchg, pid, 'HOT'])
    return code


def calc_target_pos(curr_pos: Dict, target_pos: Dict) -> Dict:
    """
    计算目标持仓
    :paramter curr_pos  当前持仓
    :parameter target_pos 目标持仓
    """
    for stdCode in curr_pos:
        if stdCode not in target_pos:
            target_pos[stdCode] = 0

    return target_pos


def get_abspath(filename: str):
    """
    : rootdir: The root directory
    : filename: The filename
    """
    import os
    import re

    rootdir = re.findall(r'.*?/QUANTHUB', os.getcwd())[0]

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == filename:

                # 打印找到的文件路径
                return os.path.join(subdir, file)


def save_model(model, filename):
    """
    保存模型到本地文件

    Parameters:
    - model: 要保存的机器学习模型对象
    - filename: 要保存的文件名，通常以 .pkl 为扩展名
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"模型已保存至 {filename}")


def load_model(filename):
    """
    从本地文件加载机器学习模型

    Parameters:
    - filename: 要加载的文件名，通常以 .pkl 为扩展名

    Returns:
    - loaded_model: 加载的机器学习模型对象
    """
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"模型已从 {filename} 加载成功")
    return loaded_model


def get_class_and_method_names(module, filter_str):
    """
    获取一个模块的所有类或者方法名，名称中包含某字符作为过滤条件。

    参数：
        module:要获取类和方法名的模块。
        filter_str,过滤条件,类或方法名中必须包含此字符。

    返回：
        一个包含类和方法名的列表。
    """

    # 获取模块中的所有类和方法。
    classes_and_methods = inspect.getmembers(
        module, inspect.isclass) + inspect.getmembers(module, inspect.isfunction)

    # 过滤出名称中包含指定字符的类和方法。
    filtered_classes_and_methods = {
        name: obj for name, obj in classes_and_methods if filter_str in name}

    # 返回类和方法名。
    return filtered_classes_and_methods


def copy_files(src_dir, dst_dir, file_names):
    """
    將檔案從一個目錄複製到另一個目錄。

    Args:
        src_dir (str): 來源目錄。
        dst_dir (str): 目標目錄。
        file_names (list): 要複製的檔案名稱清單。
    """

    # 檢查來源和目標目錄是否存在
    if not os.path.exists(src_dir):
        raise ValueError("來源目錄不存在：{}".format(src_dir))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍歷檔案名稱
    for file_name in file_names:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)

        # 檢查檔案是否存在於來源目錄
        if not os.path.isfile(src_file):
            raise ValueError("檔案不存在於來源目錄：{}".format(src_file))

        # 將檔案複製到目標目錄
        shutil.copy2(src_file, dst_file)
        print(f'{file_name} from "{src_dir}" to "{dst_dir}".')

# %%

# if __name__ == '__main__':

#     src_file_path = '/home/QUANTHUB/demos/test_hotpicker/'
#     dis_file_path = '/home/QUANTHUB/demos/common/'

#     copy_files(src_file_path,dis_file_path,['hots.json','hotmap.json'])
