from pathlib import Path
import sys

# The path to the requirements.txt file
requirements_path = Path(__file__).parents[1] / "requirements.txt"


def get_missing_requirements_message():
    """The warning message to display when a package is missing."""

    extra = ""
    if sys.flags.no_user_site:
        extra = "-s "
    return f"""
Please install the updated requirements.txt file by running:
{sys.executable} {extra}-m pip install -r {requirements_path}
If you are on the portable package you can run: update\\update_comfyui.bat to solve this problem.
""".strip()
