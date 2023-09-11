def ui_signal(signals:str|list[str]):
    """
    Return a decorator for Node classes.
    @param signals - a list of strings that name the signals to be sent to the UI. 
    (For convenience, a string gets converted to a list of length 1)

    The decorator performs the following:
    The class has OUTPUT_NODE set to True.
    The class UI_OUTPUT is appended (or created) with a comma separated list of these signals
    The class FUNCTION is wrapped such that the last len(signals) are removed, and added to the
    ui dictionary using signals as keys.

    So ui_signals(["first","second"]) will wrap a function returning (something, somethingelse, first_signal, second_signal)
    and will return { "ui": {"first":first_signal, "second":second_signal}, "result":(something, somethingelse) }
    """
    signals:iter = [signals,] if isinstance(signals,str) else signals
    def decorator(clazz):
        internal_function_name = getattr(clazz,'FUNCTION')
        if internal_function_name=='_ui_signal_decorated_function':
            raise Exception("Can't nest ui_signal decorators")
        def _ui_signal_decorated_function(self, **kwargs):
            returns = getattr(self,internal_function_name)(**kwargs)
            returns_tuple = returns['result']    if isinstance(returns,dict) else returns
            returns_ui    = returns.get('ui',{}) if isinstance(returns,dict) else {}

            popped_returns = returns_tuple[-len(signals):]
            returns_tuple  = returns_tuple[:-len(signals)]

            for i,key in enumerate(signals):
                returns_ui[key] = popped_returns[i]

            return { "ui":returns_ui, "result": returns_tuple }
        clazz._ui_signal_decorated_function = _ui_signal_decorated_function
        clazz.FUNCTION = '_ui_signal_decorated_function'
        clazz.OUTPUT_NODE = True
        clazz.UI_OUTPUT = clazz.UI_OUTPUT+"," if hasattr(clazz, 'UI_OUTPUT') else ""
        clazz.UI_OUTPUT += ",".join(signals)
        return clazz

    return decorator
        
            
