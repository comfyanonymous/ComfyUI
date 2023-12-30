

import logging
import sentry_sdk

from config.config import CONFIG


class LogUtils:
    
    @staticmethod
    def visible_convert(val, max_len=500):
        """
        Convert data into a visually friendly string
        
        INPUT:
            val, the logging data
            max_len, the max length of the __str__ data that need to convert
            
        OUTPUT:
            str or list or dict, final logging data
        """
        if isinstance(val, str):
            if len(val) > max_len:
                return 'type = str, data = HIDDEN(string too long).'
            else:
                return val
        elif isinstance(val, list):
            new_data = []
            for i, vali in enumerate(val):
                new_data.append(LogUtils.visible_convert(vali, max_len))
            return new_data
        
        elif isinstance(val, tuple):
            new_data = []
            for i, vali in enumerate(val):
                new_data.append(LogUtils.visible_convert(vali, max_len))
            new_data = tuple(new_data)
            return new_data
    
        elif isinstance(val, dict):
            new_data = {}
            for key, vali in val.items():
                new_data[key] = LogUtils.visible_convert(vali, max_len)
            return new_data
        
        else:
            val_str = val.__str__()
            if len(val_str) > max_len:
                return f'type = {type(val)}, data = HIDDEN(string too long).'
            else:
                return val
        
        return val
        





class AppLog:
    """
    logger that use logging package
    """

    

    @staticmethod
    def _get_log_level():
        """
        Get log level
        """
        log_level_str = CONFIG['log']['log_level']
        if log_level_str == 'DEBUG':
            return logging.DEBUG
        if log_level_str == 'INFO':
            return logging.INFO
        if log_level_str == 'WARNING':
            return logging.WARNING
        if log_level_str == 'ERROR':
            return logging.ERROR
        if log_level_str == 'CRITICAL':
            return logging.CRITICAL
        
        return logging.INFO



    @staticmethod
    def init():
        # init logging config
        if CONFIG['log']['console_log']:
            logging.basicConfig(
                level=AppLog._get_log_level(),  
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' 
            )
        else:
            logging.basicConfig(
                filename=CONFIG['log']['log_file'], 
                level=AppLog._get_log_level(),  
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' 
            )

        if 'sentry' in CONFIG:
            sentry_sdk.init(
                dsn=CONFIG['sentry']['dsn'],
                # Set traces_sample_rate to 1.0 to capture 100%
                # of transactions for performance monitoring.
                # We recommend adjusting this value in production.
                traces_sample_rate=1.0,
                # Set profiles_sample_rate to 1.0 to profile 100%
                # of sampled transactions.
                # We recommend adjusting this value in production.
                profiles_sample_rate=1.0,
            )



    @staticmethod
    def debug(msg):
        logging.debug(msg)


    @staticmethod
    def info(msg):
        logging.info(msg)

    @staticmethod
    def warning(msg):
        logging.warning(msg)


    @staticmethod
    def warning_report(msg):
        logging.warning(msg)
        sentry_sdk.capture_message(f'[AIGCTask]{msg}')
        

    @staticmethod
    def error(msg):
        logging.error(msg)
        # SMTPReporter.report_error(msg)
        sentry_sdk.capture_message(f'[AIGCTask]{msg}')


    @staticmethod
    def critical(msg):
        logging.critical(msg)
        # email report
        # SMTPReporter.report_error(msg)
        sentry_sdk.capture_message(f'[AIGCTask]{msg}')


    @staticmethod
    def need_debug():
        log_level = AppLog._get_log_level()
        if log_level is logging.DEBUG:
            return True
        return False

    @staticmethod
    def need_info():
        log_level = AppLog._get_log_level()
        if log_level is logging.DEBUG or log_level is logging.INFO:
            return True
        return False
    
    @staticmethod
    def need_warning():
        log_level = AppLog._get_log_level()
        if log_level is logging.WARNING or log_level is logging.INFO or log_level is logging.DEBUG:
            return True
        return False
    
    @staticmethod
    def need_error():
        log_level = AppLog._get_log_level()
        if log_level is not logging.CRITICAL:
            return True
        return False
    

    @staticmethod
    def need_critical():
        return True



    @staticmethod
    def visible_convert(val, max_len=500):
        return LogUtils.visible_convert(val, max_len)
    
    
    