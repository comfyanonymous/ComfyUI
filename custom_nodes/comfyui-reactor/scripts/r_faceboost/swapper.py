import cv2
import numpy as np

# The following code is almost entirely copied from INSwapper; the only change here is that we want to use Lanczos
# interpolation for the warpAffine call. Now that the face has been restored, Lanczos represents a good compromise
# whether the restored face needs to be upscaled or downscaled.
def in_swap(img, bgr_fake, M):
    target_img = img
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)

    # Note the use of bicubic here; this is functionally the only change from the source code
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0, flags=cv2.INTER_CUBIC)

    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white[img_white > 20] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    # k = max(mask_size//20, 6)
    # k = 6
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    k = max(mask_size // 20, 5)
    # k = 3
    # k = 3
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask /= 255
    # img_mask = fake_diff
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    fake_merged = fake_merged.astype(np.uint8)
    return fake_merged
