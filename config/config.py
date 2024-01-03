


import sys
import json
from comfy.cli_args import args

class ServiceConfig:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)


            # load config file
            cls._instance.all_configs = {}
            cls._instance._load_config()


            cls._instance.which_config = args.config
            print(cls._instance.which_config)

            assert cls._instance.which_config in cls._instance.all_configs
            cls._instance.config = cls._instance.all_configs.get(cls._instance.which_config, {})
        return cls._instance
    

    def _load_config(self):
        with open('config/config.json', 'r', encoding='utf-8') as config_file:
            self.all_configs = json.loads(config_file.read())

    def __getitem__(self, key):
        return self.config[key]
    
    def __contains__(self, key):
        return key in self.config


CONFIG = ServiceConfig()
