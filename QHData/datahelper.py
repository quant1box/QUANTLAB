# %%
import os
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient, InsertOne
import logging
from typing import List
try:
    from wtpy.wrapper import WtDataHelper

    dtHelper = WtDataHelper()
except Exception as e:
    print(e)

# %%
# datahelper = WtDataHelper()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(
            "mongodb_import.log"), logging.StreamHandler()]
    )


def chunkify(data, chunk_size):
    """将数据分成大小为 chunk_size 的块"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def batch_insert_data(collection, data, batch_size):
    """分批插入数据"""
    chunks = chunkify(data, batch_size)
    for chunk in chunks:
        requests = [InsertOne(doc) for doc in chunk]
        try:
            result = collection.bulk_write(requests, ordered=False)
            # print(f"Successfully inserted {result.inserted_count} documents.")
        except Exception as e:
            continue
            # print(f"Error during bulk write: {str(e)}")


def store_text_to_mongodb(folder_path: str,
                          database_name: str = 'multicharts',
                          collection_name: str = 'future_min'
                          ) -> None:
    """
    将指定文件夹下的所有txt文件内容存储到MongoDB中,并为date、time、code字段创建联合索引。

    Parameters:
        - folder_path (str): 要遍历的文件夹路径。
        - database_name (str): MongoDB数据库名称。
        - collection_name (str): MongoDB集合名称。

    Returns:
        None
    """

    try:
        # 连接到本地MongoDB服务器
        client = MongoClient('mongodb://192.168.31.220:27017/')

        # 创建或选择数据库
        db = client[database_name]

        # 创建或选择集合
        collection = db[collection_name]

        # 为date、time、code字段创建联合索引
        collection.create_index(
            [('datetime', 1), ('code', 1)], unique=True)

        # 遍历文件夹下的所有txt文件,使用tqdm添加进度条
        files = [f for f in os.listdir(folder_path) if os.path.isfile(
            os.path.join(folder_path, f))]

        # 使用tqdm 迭代器并设置total
        with tqdm(total=len(files), desc='Processing files', unit='file') as pbar:
            for filename in os.listdir(folder_path):

                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)

                    # 使用pandas读取文件内容
                    columns = ['date', 'time', 'open',
                               'high', 'low', 'close', 'volume']

                    try:
                        df = pd.read_csv(file_path, header=0, names=columns)
                        df['code'] = filename.split(' ')[0]
                        df['type'] = 'm5'
                        df['datetime'] = pd.to_datetime(
                            df['date'] + ' ' + df['time'])
                        df['datetime'] = df['datetime'].map(
                            lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

                        df.drop(['date', 'time'], axis=1, inplace=True)

                        dt = df.to_dict(orient='records')

                        # 分批插入数据
                        batch_insert_data(
                            collection=collection, data=dt, batch_size=1000)

                    except Exception as e:
                        continue
                        # 处理读取文件或插入数据库时的异常
                        # logging.error(
                        # f"Error processing file {filename}: {str(e)}")

                    pbar.update(1)

    except Exception as e:
        # 处理连接数据库时的异常
        logging.error(f"Error connecting to MongoDB: {str(e)}")
    finally:
        # 关闭MongoDB连接
        client.close()


def export_futures_data_to_csv(database_name: str = 'multicharts',
                               collection_name: str = 'future_min',
                               output_folder: str = 'storage/csv'):
    """
    从MongoDB读取期货数据,按商品code分组,将每组数据保存为CSV文件。

    Parameters:
        - database_name (str): MongoDB数据库名称。
        - collection_name (str): MongoDB集合名称。
        - output_folder (str): 保存CSV文件的文件夹路径。

    Returns:
        None
    """
    try:
        # 连接到MongoDB服务器
        client = MongoClient('mongodb://192.168.31.220:27017/')

        # 创建或选择数据库
        db = client[database_name]

        # 创建或选择集合
        collection = db[collection_name]

        # 获取所有不同的商品code
        distinct_codes = collection.distinct('code')

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 遍历每个商品code，将对应数据保存为CSV文件
        for code in distinct_codes:
            # 查询该商品code对应的数据
            query = {'code': code}
            cursor = collection.find(query).sort([('date', 1), ('time', 1)])

            # 将数据转换为DataFrame
            df = pd.DataFrame(list(cursor))
            df[['date', 'time']] = df['datetime'].str.split(expand=True)

            period = df.type.unique()[0]
            code = df.code.unique()[0]

            df.drop(['_id', 'code', 'type', 'datetime'], axis=1, inplace=True)
            df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]

            # 构建输出文件路径
            file_name = code + '.HOT' + f'_{period}'
            output_file_path = os.path.join(output_folder, f'{file_name}.csv')

            # 保存DataFrame为CSV文件
            df.to_csv(output_file_path, index=False)

            print(f"Saved data for code {code} to {output_file_path}")

    except Exception as e:
        # 处理异常
        print(f"Error: {str(e)}")
    finally:
        # 关闭MongoDB连接
        client.close()


def parse_and_store_dsb_files(folder_path, database_name: str = 'WT', server_url='mongodb://localhost:27017/'):
    """
    解析一个文件夹下所有.dsb文件，将数据存储到本地MongoDB数据库。

    Parameters:
        - folder_path (str): 要遍历的文件夹路径。
        - database_name (str): MongoDB数据库名称。
        - server_url (str): MongoDB服务器URL，默认为本地服务器。

    Returns:
        None
    """
    try:
        # 连接到MongoDB服务器
        client = MongoClient(server_url)

        # 创建或选择数据库
        db = client[database_name]

        # 确保文件夹路径存在
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} not found.")
            return

        # 获取所有.dsb文件
        dsb_files = [f for f in os.listdir(folder_path) if f.endswith('.dsb')]

        # 遍历所有.dsb文件
        for dsb_file in dsb_files:
            file_path = os.path.join(folder_path, dsb_file)

            # 使用pandas读取.dsb文件
            df = pd.read_dask_bag(file_path).compute()

            # 获取文件名（不含扩展名）作为集合名称
            collection_name = os.path.splitext(dsb_file)[0]

            # 存储数据到MongoDB
            db[collection_name].insert_many(df.to_dict(orient='records'))

            print(
                f"Data from {dsb_file} stored in MongoDB collection: {collection_name}")

    except Exception as e:
        # 处理异常
        print(f"Error: {str(e)}")
    finally:
        # 关闭MongoDB连接
        client.close()


def dsb2csv(dsb_file_path: str, date_range: List[str]):
    """
    將wondertrader使用的dsb文件轉換為csv文件.

    :param dsb_file_path: dsb文件根目錄
    :param date_range: 日期範圍列表
    :param datahelper: 数据助手实例,用于执行dsb到csv的转换
    """
    matching_folders = []

    # 遍历根文件夹及其子文件夹
    for dirpath, dirnames, filenames in os.walk(dsb_file_path):
        for dirname in dirnames:
            for d in date_range:
                if d in dirname:
                    # 如果文件夹名包含指定日期字符串，则将其完整路径添加到匹配列表中
                    matching_folders.append(os.path.join(dirpath, dirname))

    for dsbfolder in matching_folders:
        csvfolder = dsbfolder.replace('ticks', 'csv')

        # 如果csv文件夹不存在就创建它
        if not os.path.exists(csvfolder):
            os.makedirs(csvfolder)

        # 执行dsb到csv的转换
        datahelper.dump_ticks(binFolder=dsbfolder, csvFolder=csvfolder)


def csv2mongo(folder_path: str, db_name: str, collection_name: str, mongo_uri: str = 'mongodb://localhost:27017/'):
    """
    遍历文件夹中的所有 CSV 文件，并保存到 MongoDB 数据库中。

    :param folder_path: 包含 CSV 文件的文件夹路径
    :param db_name: MongoDB 数据库名称
    :param collection_name: MongoDB 集合名称
    :param mongo_uri: MongoDB 连接 URI
    """
    # 连接到 MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # 遍历文件夹中的所有 CSV 文件
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                # 读取 CSV 文件
                data = pd.read_csv(file_path)
                # 将 DataFrame 转换为字典列表
                records = data.to_dict(orient='records')

                batch_insert_data(collection=collection,
                                  data=records, batch_size=1000)


def his_dsb_bars():
    """
    """
    pass


def fetch_data_hisdsb(root_dir: str, code: str):
    """"""
    data_arr = []
    for dirpath, dirname, filenames in os.walk(root_dir):

        for filename in filenames:
            if filename.startswith(code):

                tick_data = dtHelper.read_dsb_ticks(
                    os.path.join(dirpath, filename)).to_df()
                data_arr.append(tick_data)

    df = pd.concat(data_arr, axis=0)
    drop_cols = ['bid_price_6', 'bid_price_7', 'bid_price_8', 'bid_price_9',
                 'ask_price_6', 'ask_price_7', 'ask_price_8', 'ask_price_9',
                 'bid_qty_6', 'bid_qty_7', 'bid_qty_8', 'bid_qty_9',
                 'ask_qty_6', 'ask_qty_7', 'ask_qty_8', 'ask_qty_9',
                 'action_date', 'action_time'
                 ]

    df.drop(drop_cols, axis=1, inplace=True)

    return df

# # 指定要遍历的文件夹路径和数据库信息
# if __name__ == '__main__':

    # export_futures_data_to_csv(output_folder='storage/csv')

    # path = '/Users/chiang/storage/csv'
    # store_text_to_mongodb(folder_path=path)
