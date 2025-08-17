COLORS_FG = {
  'BLACK': '\33[30m',
  'RED': '\33[31m',
  'GREEN': '\33[32m',
  'YELLOW': '\33[33m',
  'BLUE': '\33[34m',
  'MAGENTA': '\33[35m',
  'CYAN': '\33[36m',
  'WHITE': '\33[37m',
  'GREY': '\33[90m',
  'BRIGHT_RED': '\33[91m',
  'BRIGHT_GREEN': '\33[92m',
  'BRIGHT_YELLOW': '\33[93m',
  'BRIGHT_BLUE': '\33[94m',
  'BRIGHT_MAGENTA': '\33[95m',
  'BRIGHT_CYAN': '\33[96m',
  'BRIGHT_WHITE': '\33[97m',
}
COLORS_STYLE = {
  'RESET': '\33[0m',
  'BOLD': '\33[1m',
  'NORMAL': '\33[22m',
  'ITALIC': '\33[3m',
  'UNDERLINE': '\33[4m',
  'BLINK': '\33[5m',
  'BLINK2': '\33[6m',
  'SELECTED': '\33[7m',
}
COLORS_BG = {
  'BLACK': '\33[40m',
  'RED': '\33[41m',
  'GREEN': '\33[42m',
  'YELLOW': '\33[43m',
  'BLUE': '\33[44m',
  'MAGENTA': '\33[45m',
  'CYAN': '\33[46m',
  'WHITE': '\33[47m',
  'GREY': '\33[100m',
  'BRIGHT_RED': '\33[101m',
  'BRIGHT_GREEN': '\33[102m',
  'BRIGHT_YELLOW': '\33[103m',
  'BRIGHT_BLUE': '\33[104m',
  'BRIGHT_MAGENTA': '\33[105m',
  'BRIGHT_CYAN': '\33[106m',
  'BRIGHT_WHITE': '\33[107m',
}

def log_node_success(node_name, message=None):
  """Logs a success message."""
  _log_node(COLORS_FG["GREEN"], node_name, message)

def log_node_info(node_name, message=None):
  """Logs an info message."""
  _log_node(COLORS_FG["CYAN"], node_name, message)


def log_node_warn(node_name, message=None):
  """Logs an warn message."""
  _log_node(COLORS_FG["YELLOW"], node_name, message)

def log_node_error(node_name, message=None):
  """Logs an warn message."""
  _log_node(COLORS_FG["RED"], node_name, message)

def log_node(node_name, message=None):
  """Logs a message."""
  _log_node(COLORS_FG["CYAN"], node_name, message)


def _log_node(color, node_name, message=None, prefix=''):
  print(_get_log_msg(color, node_name, message, prefix=prefix))

def _get_log_msg(color, node_name, message=None, prefix=''):
  msg = f'{COLORS_STYLE["BOLD"]}{color}{prefix}[EasyUse] {node_name.replace(" (EasyUse)", "")}'
  msg += f':{COLORS_STYLE["RESET"]} {message}' if message is not None else f'{COLORS_STYLE["RESET"]}'
  return msg

