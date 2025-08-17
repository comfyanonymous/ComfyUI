# TODO: In case anyone that knows how to set up PyTest correctly comes around, this file can be scrapped.
from pathlib import Path

TEXT_TYPE = "STRING"

CLASS_NAME = "WAS_Text_Sort"
class_string = f"class {CLASS_NAME}:"
exec(class_string + Path("../WAS_Node_Suite.py").read_text().split(class_string)[1].split("class ")[0])

def was_text_sort(text = "", separator = WAS_Text_Sort.INPUT_TYPES()["required"]["separator"][1]["default"]):
    return WAS_Text_Sort().sort(text, separator)[0]
