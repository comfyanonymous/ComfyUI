
import io, os

import base64
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


"""
Image util funcitons.
"""




class ImageUtil:

    @staticmethod
    def get_pil_metadata(pil_image):
        # Copy any text-only metadata
        metadata = PngImagePlugin.PngInfo()
        for key, value in pil_image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)

        return metadata



    @staticmethod
    def image_to_base64(img: Image, format='PNG') -> str:
        """Converts a PIL Image object to a base64 encoded string.

        Args:
            img (PIL.Image): The image object to convert.

        Returns:
            str: The base64 encoded string.
        """
        # Convert the image to bytes using an in-memory buffer
        buffer = io.BytesIO()
        if format == 'PNG':
            img.save(buffer, format=format, pnginfo=ImageUtil.get_pil_metadata(img))
        else:
            img.save(buffer, format=format)
        img_bytes = buffer.getvalue()

        # Encode the bytes as base64
        base64_str = base64.b64encode(img_bytes).decode()

        return base64_str
    

    

    @staticmethod
    def base64_to_image(base64_str: str) -> Image:
        """Converts a base64 encoded string to a PIL Image object.

        Args:
            base64_str (str): The base64 encoded string.

        Returns:
            PIL.Image: The image object.
        """
        # Decode the base64 string to bytes
        img_bytes = base64.b64decode(base64_str)

        # Convert the bytes to a PIL Image object
        img = Image.open(io.BytesIO(img_bytes))

        return img
    


    @staticmethod
    def draw_text_on_image(image, text, font_path, font_size, color, center_pos):
        """
        Draw text on the given Image object.

        Args:
        image (PIL.Image): The Image object to draw on
        text (str): The text to be drawn
        font_path (str): The path of font
        font_size (int): Font size
        color (tuple): Color of the text, e.g., (255, 255, 255) for white.
        center_pos (tuple): The center position where the text will be drawn, e.g., (100, 200)

        Returns:
        PIL.Image
        """
        # create a drawable object
        draw = ImageDraw.Draw(image)

        # load the font
        font = ImageFont.truetype(font_path, font_size)
        # text size
        text_width, text_height = draw.textsize(text, font=font)
        # text position
        pos_x = center_pos[0] - (text_width) / 2
        pos_y = center_pos[1] - (text_height) / 2 

        # draw the text
        draw.text((pos_x, pos_y), text, font=font, fill=color)

        return image
    

    @staticmethod
    def save_image_with_create_dir(image, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(path)



    @staticmethod
    def resize_image(resize_mode, im:Image, width, height):
        """
        Resizes an image with the specified resize_mode, width, and height.

        Args:
            resize_mode: The mode to use when resizing the image.
                0: Resize the image to the specified width and height.
                1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
                2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
            im: The image to resize.
            width: The width to resize the image to.
            height: The height to resize the image to.
        """


        if resize_mode == 0:
            res = im.resize((width, height))

        elif resize_mode == 1:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio > src_ratio else im.width * height // im.height
            src_h = height if ratio <= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h))
            res = Image.new("RGB", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        else:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio < src_ratio else im.width * height // im.height
            src_h = height if ratio >= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h))
            res = Image.new("RGB", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                if fill_height > 0:
                    res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                    res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                if fill_width > 0:
                    res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                    res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

        return res
        

