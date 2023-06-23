
class LatentFormat:
    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor

class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor

class SDXL(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.13025

