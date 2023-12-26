

import os
from io import BytesIO

import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo


import folder_paths
from config.config import CONFIG


def get_dir_by_type(dir_type):
    if dir_type is None:
        dir_type = "input"

    if dir_type == "input":
        type_dir = folder_paths.get_input_directory()
    elif dir_type == "temp":
        type_dir = folder_paths.get_temp_directory()
    elif dir_type == "output":
        type_dir = folder_paths.get_output_directory()

    return type_dir, dir_type




class LocalResourceLoader:
    """
    Load or Upload resource from local device.
    """

    @staticmethod
    def image_upload(post, image_save_function=None):
        image = post.get("image")
        overwrite = post.get("overwrite")

        image_upload_type = post.get("type")
        upload_dir, image_upload_type = get_dir_by_type(image_upload_type)

        print("image is: ")
        print(image)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        if image and image.file:
            filename = image.filename
            if not filename:
                return False, None

            subfolder = post.get("subfolder", "")
            full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
            filepath = os.path.abspath(os.path.join(full_output_folder, filename))

            if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                return False, None

            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            split = os.path.splitext(filename)

            if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                pass
            else:
                i = 1
                while os.path.exists(filepath):
                    filename = f"{split[0]} ({i}){split[1]}"
                    filepath = os.path.join(full_output_folder, filename)
                    i += 1

            print(image_save_function)
            if image_save_function is not None:
                succ = image_save_function(image, post, filepath)
            else:
                with open(filepath, "wb") as f:
                    f.write(image.file.read())
                succ = True
                    
            if not succ:
                return False, None

            return True, {"name" : filename, "subfolder": subfolder, "type": image_upload_type}
        else:
            return False, None
        
        
    @staticmethod
    def mask_save_function(image, post, filepath):
        original_ref = json.loads(post.get("original_ref"))
        filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

        # validation for security: prevent accessing arbitrary path
        if filename[0] == '/' or '..' in filename:
            return False

        if output_dir is None:
            type = original_ref.get("type", "output")
            output_dir = folder_paths.get_directory_by_type(type)

        if output_dir is None:
            return False

        if original_ref.get("subfolder", "") != "":
            full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
            if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                return False
            output_dir = full_output_dir

        file = os.path.join(output_dir, filename)

        if os.path.isfile(file):
            with Image.open(file) as original_pil:
                metadata = PngInfo()
                if hasattr(original_pil,'text'):
                    for key in original_pil.text:
                        metadata.add_text(key, original_pil.text[key])
                original_pil = original_pil.convert('RGBA')
                mask_pil = Image.open(image.file).convert('RGBA')

                # alpha copy
                new_alpha = mask_pil.getchannel('A')
                original_pil.putalpha(new_alpha)
                original_pil.save(filepath, compress_level=4, pnginfo=metadata)
                
                
                
    @staticmethod
    def mask_upload(post):
        """
        upload mask locally
        """
        return LocalResourceLoader.image_upload(post, LocalResourceLoader.mask_save_function)
        




class ServerResourceLoader:
    """
    Load or upload resource from server object storage.
    """
    
    @staticmethod
    def image_upload(post):
        pass
    
    @staticmethod
    def mask_upload(post):
        pass





class Resourceloader:
    
    @staticmethod
    def image_upload(post):
        if CONFIG["deploy"]:
            return ServerResourceLoader.image_upload(post)
        else:
            return LocalResourceLoader.image_upload(post)
    
    @staticmethod
    def mask_upload(post):
        if CONFIG["deploy"]:
            return ServerResourceLoader.mask_upload(post)
        else:
            return LocalResourceLoader.mask_upload(post)
    

