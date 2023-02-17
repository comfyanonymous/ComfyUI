import cv2


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold, l2gradient):
        return cv2.Canny(img, low_threshold, high_threshold, L2gradient=l2gradient)
