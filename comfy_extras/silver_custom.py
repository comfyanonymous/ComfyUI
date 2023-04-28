import cv2
import torch

class ExpandImageMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE", )
                }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("IMAGE", "MASK", )
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

            # Invert binary mask image
            # im_bw = cv2.bitwise_not(im_bw)

            # Convert binary mask image to PyTorch tensor
            img = torch.from_numpy(im_bw).unsqueeze(0).float()

            # Append mask image tensor to list
            mask_images.append(img)

        # Stack list of mask image tensors into a single tensor
        mask_images_tensor = torch.cat(mask_images)

        # Return tuple of mask images and single mask image
        single_mask_image = mask_images_tensor[0, :, :]
        return mask_images_tensor, single_mask_image


NODE_CLASS_MAPPINGS = {
    "ExpandImageMask": ExpandImageMask
}
