from PIL import Image, ImageFile, UnidentifiedImageError

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

def open_image(path):
    prev_value = None

    try:
        img = Image.open(path)
    except (UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(path)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
        return img
