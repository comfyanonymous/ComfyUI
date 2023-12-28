
import pymongo

from config.config import CONFIG



class MongoConnection:
    """
    Singleton class.
    Manage the connection to MongoDB.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            cls._instance.connection = None
            cls._instance.db = None

        return cls._instance


    def __del__(self):
        self.connection = None
        self.db = None
        MongoConnection._instance = None
    

    def get_db(self):
        # connect
        if self.db is None:
            db_setting = CONFIG['mongodb_settings']
            if self.connection is None:
                url = db_setting['url']
                self.connection = pymongo.MongoClient(url)
            db_name =db_setting['database']
            self.db = self.connection[db_name]
            
        return self.db



    def get_table(self, table_name):
        _db = self.get_db()
        return _db[CONFIG['mongodb_settings'][table_name]]

    
    