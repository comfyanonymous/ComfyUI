




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
        
