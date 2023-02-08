

current_loaded_model = None


def unload_model():
    global current_loaded_model
    if current_loaded_model is not None:
        current_loaded_model.model.cpu()
        current_loaded_model.unpatch_model()
        current_loaded_model = None


def load_model_gpu(model):
    global current_loaded_model
    if model is current_loaded_model:
        return
    unload_model()
    try:
        real_model = model.patch_model()
    except Exception as e:
        model.unpatch_model()
        raise e
    current_loaded_model = model
    real_model.cuda()
    return current_loaded_model
