import PIL
import numpy as np
import cv2
import torch
from PIL.Image import Image

class ExpandImageMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE", )
                }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK", )
    FUNCTION = "image_to_mask_image"

    def image_to_mask_image(self, images):
        mask_images = []
        for image in images:
            i = 255. * image.cpu().numpy()

            # Convert to grayscale
            image_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

            # Apply blurring to grayscale image
            image_gray = cv2.blur(image_gray, (10, 10))
            image_gray = cv2.blur(image_gray, (20, 20))

            # Convert image to the expected data type
            image_gray = cv2.convertScaleAbs(image_gray)

            # Apply threshold to grayscale image
            (thresh, im_bw) = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Apply blurring to binary mask image
            ksize = (50, 50)
            im_bw = cv2.blur(im_bw, ksize)

            # Threshold binary mask image again
            im_bw = cv2.threshold(im_bw, thresh, 255, cv2.THRESH_BINARY)[1]

            # Convert binary mask image to 3-channel RGB image
            mask_image_rgb = np.zeros_like(i)
            mask_image_rgb[:, :, 0] = im_bw
            mask_image_rgb[:, :, 1] = im_bw
            mask_image_rgb[:, :, 2] = im_bw
            pil_image = PIL.Image.fromarray(np.uint8(mask_image_rgb))

            # create a new alpha channel with all pixels set to 255 (fully opaque)
            alpha = PIL.Image.new('L', pil_image.size, 255)

            # iterate over each pixel and set the alpha channel to 0 if the RGB values are white
            for x in range(pil_image.width):
                for y in range(pil_image.height):
                    if pil_image.getpixel((x, y)) == (255, 255, 255):
                        alpha.putpixel((x, y), 0)

            # merge the alpha channel with the original image
            pil_image.putalpha(alpha)

            # Append mask image tensor to list
            mask_images.append(1. - torch.from_numpy(np.array(pil_image.getchannel('A')).astype(np.float32) / 255.0))

        return mask_images


NODE_CLASS_MAPPINGS = {
    "ExpandImageMask": ExpandImageMask
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpandImageMask": "Expand Image Mask"
}
