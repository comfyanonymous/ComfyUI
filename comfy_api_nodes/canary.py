import av

ver = av.__version__.split(".")
if int(ver[0]) < 14:
    raise Exception("INSTALL NEW VERSION OF PYAV TO USE API NODES.")

if int(ver[0]) == 14 and int(ver[1]) < 2:
    raise Exception("INSTALL NEW VERSION OF PYAV TO USE API NODES.")

NODE_CLASS_MAPPINGS = {}
