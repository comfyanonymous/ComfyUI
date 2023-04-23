import cv2
import torch

class ExpandImageMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",)
                }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_to_mask_image"

    def image_to_mask_image(self, images):
        mask_images = []
        for image in images:
            i = 255. * image.cpu().numpy()
            # opencv_image = PIL.Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # cv2.imwrite('opencv_image.png', i)

            image_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            image_gray = cv2.blur(image_gray, (10, 10))
            # cv2.imwrite('image_gray1.png', image_gray)
            image_gray = cv2.blur(image_gray, (20, 20))
            # cv2.imwrite('image_gray2.png', image_gray)

            # Convert the image to the expected data type
            image_gray = cv2.convertScaleAbs(image_gray)

            # Apply the threshold using the modified image
            (thresh, im_bw) = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            ksize = (50, 50)
            im_bw = cv2.blur(im_bw, ksize)
            im_bw = cv2.threshold(im_bw, thresh, 255, cv2.THRESH_BINARY)[1]
            im_bw = cv2.bitwise_not(im_bw)
            # cv2.imwrite('im_bw.png', im_bw)

            # Convert the binary mask image to a PyTorch tensor
            img = torch.from_numpy(im_bw).unsqueeze(0).float()
            mask_images.append(img)
        return tuple(mask_images)

NODE_CLASS_MAPPINGS = {
    "ExpandImageMask": ExpandImageMask
}
