import torch

class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor

class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3512,  0.2297,  0.3227],
                    [ 0.3250,  0.4974,  0.2350],
                    [-0.2829,  0.1762,  0.2721],
                    [-0.2120, -0.2616, -0.7177]
                ]
        self.taesd_decoder_name = "taesd_decoder"

class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3920,  0.4054,  0.4549],
                    [-0.2634, -0.0196,  0.0653],
                    [ 0.0568,  0.1687, -0.0755],
                    [-0.3112, -0.2359, -0.2076]
                ]
        self.taesd_decoder_name = "taesdxl_decoder"

class SDXL_Playground_2_5(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.5
        self.latents_mean = torch.tensor([-1.6574, 1.886, -1.383, 2.5155]).view(1, 4, 1, 1)
        self.latents_std = torch.tensor([8.4927, 5.9022, 6.5498, 5.2299]).view(1, 4, 1, 1)

        self.latent_rgb_factors = [
                    #   R        G        B
                    [ 0.3920,  0.4054,  0.4549],
                    [-0.2634, -0.0196,  0.0653],
                    [ 0.0568,  0.1687, -0.0755],
                    [-0.3112, -0.2359, -0.2076]
                ]
        self.taesd_decoder_name = "taesdxl_decoder"

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


class SD_X4(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.08333
        self.latent_rgb_factors = [
            [-0.2340, -0.3863, -0.3257],
            [ 0.0994,  0.0885, -0.0908],
            [-0.2833, -0.2349, -0.3741],
            [ 0.2523, -0.0055, -0.1651]
        ]

class SC_Prior(LatentFormat):
    latent_channels = 16
    def __init__(self):
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.0326, -0.0204, -0.0127],
            [-0.1592, -0.0427,  0.0216],
            [ 0.0873,  0.0638, -0.0020],
            [-0.0602,  0.0442,  0.1304],
            [ 0.0800, -0.0313, -0.1796],
            [-0.0810, -0.0638, -0.1581],
            [ 0.1791,  0.1180,  0.0967],
            [ 0.0740,  0.1416,  0.0432],
            [-0.1745, -0.1888, -0.1373],
            [ 0.2412,  0.1577,  0.0928],
            [ 0.1908,  0.0998,  0.0682],
            [ 0.0209,  0.0365, -0.0092],
            [ 0.0448, -0.0650, -0.1728],
            [-0.1658, -0.1045, -0.1308],
            [ 0.0542,  0.1545,  0.1325],
            [-0.0352, -0.1672, -0.2541]
        ]

class SC_B(LatentFormat):
    def __init__(self):
        self.scale_factor = 1.0 / 0.43
        self.latent_rgb_factors = [
            [ 0.1121,  0.2006,  0.1023],
            [-0.2093, -0.0222, -0.0195],
            [-0.3087, -0.1535,  0.0366],
            [ 0.0290, -0.1574, -0.4078]
        ]

class SD3(LatentFormat):
    latent_channels = 16
    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609
        self.latent_rgb_factors = [
            [-0.0645,  0.0177,  0.1052],
            [ 0.0028,  0.0312,  0.0650],
            [ 0.1848,  0.0762,  0.0360],
            [ 0.0944,  0.0360,  0.0889],
            [ 0.0897,  0.0506, -0.0364],
            [-0.0020,  0.1203,  0.0284],
            [ 0.0855,  0.0118,  0.0283],
            [-0.0539,  0.0658,  0.1047],
            [-0.0057,  0.0116,  0.0700],
            [-0.0412,  0.0281, -0.0039],
            [ 0.1106,  0.1171,  0.1220],
            [-0.0248,  0.0682, -0.0481],
            [ 0.0815,  0.0846,  0.1207],
            [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456],
            [-0.1418, -0.1457, -0.1259]
        ]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

class StableAudio1(LatentFormat):
    latent_channels = 64
