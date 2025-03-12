
from cmath import log
import logging,os


class watch_dog(object):

    def __init__(self,name:str='',
                 filename:str='quanthub.log',
                 level=logging.DEBUG):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)


        self.logger.handlers.clear()
        # 创建一个handler，用于写入日志文件
        # 指定文件输出路径，注意logs是个文件夹，一定要加上/，不然会导致输出路径错误，把logs变成文件名的一部分了
        log_path = os.getcwd()+"/logs/"
        # log_path = '../logs/'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logname = log_path + filename  # 指定输出的日志文件名

        # 指定utf-8格式编码，避免输出的日志文本乱码
        fh = logging.FileHandler(logname, encoding='utf-8', mode='a')
        fh.setLevel(logging.INFO)

        

        # 创建一个handler，用于将日志输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


    def info(self, message: str):
        self.logger.info(message)

    def warn(self, message: str):
        self.logger.warn(message)

    def error(self, message: str):
        self.logger.error(message)

    def fatal(self, message: str):
        self.logger.fatal(message)
