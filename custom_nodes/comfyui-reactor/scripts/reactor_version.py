app_title = "ReActor Node for ComfyUI"
version_flag = "v0.6.1-b3"

COLORS = {
    "CYAN": "\033[0;36m",  # CYAN
    "ORANGE": "\033[38;5;173m",  # Calm ORANGE
    "GREEN": "\033[0;32m",  # GREEN
    "YELLOW": "\033[0;33m",  # YELLOW
    "RED": "\033[0;91m",  # RED
    "0": "\033[0m",  # RESET COLOR
}

print(f"\n{COLORS['YELLOW']}[ReActor]{COLORS['0']} - {COLORS['ORANGE']}STATUS{COLORS['0']} - {COLORS['GREEN']}Running {version_flag} in ComfyUI{COLORS['0']}")
