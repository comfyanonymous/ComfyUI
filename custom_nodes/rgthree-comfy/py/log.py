from .pyproject import NAME

# https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
# https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
COLORS = {
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
  # Styles.
  'RESET': '\33[0m',  # Note, Portainer doesn't like 00 here, so we'll use 0. Should be fine...
  'BOLD': '\33[01m',
  'NORMAL': '\33[22m',
  'ITALIC': '\33[03m',
  'UNDERLINE': '\33[04m',
  'BLINK': '\33[05m',
  'BLINK2': '\33[06m',
  'SELECTED': '\33[07m',
  # Backgrounds
  'BG_BLACK': '\33[40m',
  'BG_RED': '\33[41m',
  'BG_GREEN': '\33[42m',
  'BG_YELLOW': '\33[43m',
  'BG_BLUE': '\33[44m',
  'BG_MAGENTA': '\33[45m',
  'BG_CYAN': '\33[46m',
  'BG_WHITE': '\33[47m',
  'BG_GREY': '\33[100m',
  'BG_BRIGHT_RED': '\33[101m',
  'BG_BRIGHT_GREEN': '\33[102m',
  'BG_BRIGHT_YELLOW': '\33[103m',
  'BG_BRIGHT_BLUE': '\33[104m',
  'BG_BRIGHT_MAGENTA': '\33[105m',
  'BG_BRIGHT_CYAN': '\33[106m',
  'BG_BRIGHT_WHITE': '\33[107m',
}


def log_node_success(node_name, message, msg_color='RESET'):
  """Logs a success message."""
  _log_node("BRIGHT_GREEN", node_name, message, msg_color=msg_color)


def log_node_info(node_name, message, msg_color='RESET'):
  """Logs an info message."""
  _log_node("CYAN", node_name, message, msg_color=msg_color)


def log_node_error(node_name, message, msg_color='RESET'):
  """Logs an info message."""
  _log_node("RED", node_name, message, msg_color=msg_color)


def log_node_warn(node_name, message, msg_color='RESET'):
  """Logs an warn message."""
  _log_node("YELLOW", node_name, message, msg_color=msg_color)


def log_node(node_name, message, msg_color='RESET'):
  """Logs a message."""
  _log_node("CYAN", node_name, message, msg_color=msg_color)


def _log_node(color, node_name, message, msg_color='RESET'):
  """Logs for a node message."""
  log(message, color=color, prefix=node_name.replace(" (rgthree)", ""), msg_color=msg_color)


def log(message, color=None, msg_color=None, prefix=None):
  """Basic logging."""
  color = COLORS[color] if color is not None and color in COLORS else COLORS["BRIGHT_GREEN"]
  msg_color = COLORS[msg_color] if msg_color is not None and msg_color in COLORS else ''
  prefix = f'[{prefix}]' if prefix is not None else ''
  msg = f'{color}[{NAME}]{prefix}'
  msg += f'{msg_color} {message}{COLORS["RESET"]}'
  print(msg)
