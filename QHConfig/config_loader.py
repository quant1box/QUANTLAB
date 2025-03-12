import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化配置加载器
        
        Args:
            config_path (str): YAML配置文件的路径，默认为当前目录下的config.yaml
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Returns:
            Dict[str, Any]: 配置文件内容
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"配置文件 {self.config_path} 未找到")
            return {}
        except yaml.YAMLError as e:
            print(f"解析YAML文件时发生错误: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项的值
        
        Args:
            key (str): 配置项的键，支持嵌套键，如 'database.host'
            default (Any, optional): 默认值，如果未找到配置项则返回
        
        Returns:
            Any: 配置项的值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def get_database_config(self) -> Dict[str, Any]:
        """
        获取数据库配置
        
        Returns:
            Dict[str, Any]: 数据库配置信息
        """
        return self.config.get('database', {})

    def get_api_config(self, provider: str = None) -> Dict[str, Any]:
        """
        获取API配置
        
        Args:
            provider (str, optional): API提供商名称，如 'tushare', 'rqdata'
        
        Returns:
            Dict[str, Any]: API配置信息
        """
        api_config = self.config.get('api', {})
        return api_config.get(provider, {}) if provider else api_config

    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            Dict[str, Any]: 日志配置信息
        """
        return self.config.get('logging', {})

def main():
    # 使用示例
    config_loader = ConfigLoader('QHData/config.yaml')
    
    # 获取数据库配置
    db_host = config_loader.get('database.host', 'localhost')
    db_port = config_loader.get('database.port', 27017)
    print(f"数据库主机: {db_host}, 端口: {db_port}")
    
    # 获取API配置
    tushare_token = config_loader.get('api.tushare.token')
    print(f"Tushare Token: {tushare_token}")
    
    # 获取日志配置
    log_level = config_loader.get('logging.level', 'INFO')
    log_file = config_loader.get('logging.file', 'app.log')
    print(f"日志级别: {log_level}, 日志文件: {log_file}")

if __name__ == "__main__":
    main() 