

class ErrorCode:
    """
    错误码的定义。
    
    
    错误码是3-4位的十进制code
    * code后两位用来区分具体的错误类型，如同样是参数错误，可能是少了参数，或者参数类型不符合，等等
    * code其他几位是错误的大类，例如 3xx 是参数错误
    
    """
    # unexpected
    UNEXPECTED = 0
    
    # success
    SUCCESS = 1
    
    # parameter error
    MISSING_PARAM = 301
    INVALID_PARAM = 302
    
    
    
    
    # execution error
    EXE_UNEXP = 700
    