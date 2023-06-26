
class LatentFormat:
    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor

class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
                    #   R        G        B
                    [0.298, 0.207, 0.208],  # L1
                    [0.187, 0.286, 0.173],  # L2
                    [-0.158, 0.189, 0.264],  # L3
                    [-0.184, -0.271, -0.473],  # L4
                ]
        self.taesd_decoder_name = "taesd_decoder.pth"

class SDXL(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.13025
        self.latent_rgb_factors = [ #TODO: these are the factors for SD1.5, need to estimate new ones for SDXL
                    #   R        G        B
                    [0.298, 0.207, 0.208],  # L1
                    [0.187, 0.286, 0.173],  # L2
                    [-0.158, 0.189, 0.264],  # L3
                    [-0.184, -0.271, -0.473],  # L4
                ]
        self.taesd_decoder_name = "taesdxl_decoder.pth"
