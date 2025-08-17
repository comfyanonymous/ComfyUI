from r_basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class VideoGANModel(SRGANModel, VideoBaseModel):
    """Video GAN model.

    Use multiple inheritance.
    It will first use the functions of SRGANModel:
        init_training_settings
        setup_optimizers
        optimize_parameters
        save
    Then find functions in VideoBaseModel.
    """
