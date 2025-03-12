from typing import Dict, Any
import os

class Config:
    """QHData configuration settings"""
    
    # MongoDB settings
    MONGODB = {
        'host': '192.168.31.220',  # Default MongoDB host
        # 'host': 'localhost',
        'port': 27017,             # Default MongoDB port
        'maxPoolSize': 100,        # Connection pool max size
        'minPoolSize': 10,         # Connection pool min size
        'maxIdleTimeMS': 300000,   # Max connection idle time
    }

    # PostgreSQL settings 
    POSTGRESQL = {
        'host': '192.168.215.3',
        'port': 5432,
        'database': 'optuna',
        'user': 'postgres', #postgres
        'password': '035115'
    }

    # API Keys
    API_KEYS = {
        'tushare': '3ce4674d1ac49d9789c6c1353e7170789be05cd485aebbca74c9f51c',
    }

    # Data storage paths
    PATHS = {
        'csv_storage': 'storage/csv',
        'parquet_storage': 'storage/parquet',
    }

    # HTTP Request headers
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    }

    # Database names
    DB_NAMES = {
        'main': 'quantaxis',
        'rq': 'rq_data',
        'tq': 'tqdata',
        'wt': 'WT',
        'multicharts': 'multicharts'
    }

    @classmethod
    def get_mongodb_uri(cls, db_name: str = 'main') -> str:
        """Get MongoDB connection URI for specified database"""
        return f"mongodb://{cls.MONGODB['host']}:{cls.MONGODB['port']}/{cls.DB_NAMES.get(db_name, cls.DB_NAMES['main'])}"

    @classmethod
    def get_postgresql_uri(cls) -> str:
        """Get PostgreSQL connection URI"""
        return f"postgresql://{cls.POSTGRESQL['user']}:{cls.POSTGRESQL['password']}@{cls.POSTGRESQL['host']}:{cls.POSTGRESQL['port']}/{cls.POSTGRESQL['database']}"

    @classmethod
    def ensure_paths(cls) -> None:
        """Ensure all storage paths exist"""
        for path in cls.PATHS.values():
            os.makedirs(path, exist_ok=True)

            