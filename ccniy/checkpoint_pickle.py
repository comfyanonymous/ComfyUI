import pickle

load = pickle.load

class Empty:
    pass

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        #TODO: safe unpickle
        if module.startswith("pytorch_lightning"):
            return Empty
        return super().find_class(module, name)
