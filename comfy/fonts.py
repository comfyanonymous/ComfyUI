import logging
from pathlib import Path
from typing import Tuple, Union

from PIL.ImageFont import FreeTypeFont, ImageFont, load_default as load_default_font, truetype as load_truetype

from folder_paths import base_path


default_font_path = Path(base_path) / "fonts"

AnyFont = Union[FreeTypeFont, ImageFont]

class FontCollection(dict):
    """
    A dictionary that maps font names to PIL font objects.
    """

    def __init__(self, directory: Path = default_font_path):
        """
        Initialize the FontCollection with fonts found in the given directory (including subdirectories).

        Args:
            directory (Path): The path to the directory containing font files.
        """

        paths = [font for font in directory.rglob("*.[tT][tT][fF]") if font.is_file()]
        self.default_font_name, self.default_font = self.load_default_font()
        fonts = {
            self.default_font_name: self.default_font,
        }
        for path in paths:
            font_info = self.load_font(path)
            if font_info:
                if font_info[0] in fonts:
                    logging.warning(f"Fonts with duplicate names found: {font_info[0]}")
                fonts[font_info[0]] = font_info[1]
        super().__init__(fonts)

    @classmethod
    def load_default_font(cls) -> Tuple[str, AnyFont]:
        """
        Load the default PIL font.

        Returns:
            tuple[str, AnyFont]: The font's name and the font object.
        """

        font = load_default_font()
        family, style = None, None
        if isinstance(font, FreeTypeFont):
            family, style = font.getname()
        family = family or "Unknown"
        style = style or "Regular"
        return " ".join([family, style]), font

    @classmethod
    def load_font(cls, path: Path) -> Tuple[str, FreeTypeFont]:
        """
        Load a font and extract its name and style.

        Args:
            path (Path): The path to the font file.

        Returns:
            tuple[str, ImageFont.FreeTypeFont]: A tuple containing the font's name and the font object.

        Raises:
            OSError: If the font file could not be read.
            ValueError: If the font size is not greater than zero.
        """

        font = load_truetype(path)
        family, style = font.getname()
        family = family or "Unknown"
        style = style or "Regular"
        return " ".join([family, style]), font
