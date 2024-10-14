import torch

class LatentFormat:
    scale_factor = 1.0
    latent_channels = 4
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
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
                    [ 0.3651,  0.4232,  0.4341],
                    [-0.2533, -0.0042,  0.1068],
                    [ 0.1076,  0.1111, -0.0362],
                    [-0.3165, -0.2492, -0.2188]
                ]
        self.latent_rgb_factors_bias = [ 0.1084, -0.0175, -0.0011]

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
            [-0.0922, -0.0175,  0.0749],
            [ 0.0311,  0.0633,  0.0954],
            [ 0.1994,  0.0927,  0.0458],
            [ 0.0856,  0.0339,  0.0902],
            [ 0.0587,  0.0272, -0.0496],
            [-0.0006,  0.1104,  0.0309],
            [ 0.0978,  0.0306,  0.0427],
            [-0.0042,  0.1038,  0.1358],
            [-0.0194,  0.0020,  0.0669],
            [-0.0488,  0.0130, -0.0268],
            [ 0.0922,  0.0988,  0.0951],
            [-0.0278,  0.0524, -0.0542],
            [ 0.0332,  0.0456,  0.0895],
            [-0.0069, -0.0030, -0.0810],
            [-0.0596, -0.0465, -0.0293],
            [-0.1448, -0.1463, -0.1189]
        ]
        self.latent_rgb_factors_bias = [0.2394, 0.2135, 0.1925]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

class StableAudio1(LatentFormat):
    latent_channels = 64

class Flux(SD3):
    latent_channels = 16
    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors =[
            [-0.0346,  0.0244,  0.0681],
            [ 0.0034,  0.0210,  0.0687],
            [ 0.0275, -0.0668, -0.0433],
            [-0.0174,  0.0160,  0.0617],
            [ 0.0859,  0.0721,  0.0329],
            [ 0.0004,  0.0383,  0.0115],
            [ 0.0405,  0.0861,  0.0915],
            [-0.0236, -0.0185, -0.0259],
            [-0.0245,  0.0250,  0.1180],
            [ 0.1008,  0.0755, -0.0421],
            [-0.0515,  0.0201,  0.0011],
            [ 0.0428, -0.0012, -0.0036],
            [ 0.0817,  0.0765,  0.0749],
            [-0.1264, -0.0522, -0.1103],
            [-0.0280, -0.0881, -0.0499],
            [-0.1262, -0.0982, -0.0778]
        ]
        self.latent_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor
