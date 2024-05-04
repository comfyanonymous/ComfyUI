from PIL import Image, ImageFile

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

def open_image(path):
    try :
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(path)
    
    except:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(path)
        
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        return img
