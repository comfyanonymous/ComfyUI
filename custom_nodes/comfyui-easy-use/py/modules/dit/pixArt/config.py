"""
List of all PixArt model types / settings
"""
sampling_settings = {
	"beta_schedule" : "sqrt_linear",
	"linear_start"  : 0.0001,
	"linear_end"    : 0.02,
	"timesteps"     : 1000,
}

pixart_conf = {
	"PixArtMS_XL_2": { # models/PixArtMS
		"target": "PixArtMS",
		"unet_config": {
			"input_size"  : 1024//8,
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1152,
			"pe_interpolation": 2,
		},
		"sampling_settings" : sampling_settings,
	},
	"PixArtMS_Sigma_XL_2": {
		"target": "PixArtMSSigma",
		"unet_config": {
			"input_size"  : 1024//8,
			"token_num"   :  300,
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1152,
			"micro_condition": False,
			"pe_interpolation": 2,
			"model_max_length": 300,
		},
		"sampling_settings" : sampling_settings,
	},
	"PixArtMS_Sigma_XL_2_900M": {
		"target": "PixArtMSSigma",
		"unet_config": {
			"input_size": 1024 // 8,
			"token_num": 300,
			"depth": 42,
			"num_heads": 16,
			"patch_size": 2,
			"hidden_size": 1152,
			"micro_condition": False,
			"pe_interpolation": 2,
			"model_max_length": 300,
		},
		"sampling_settings": sampling_settings,
	},
	"PixArtMS_Sigma_XL_2_2K": {
		"target": "PixArtMSSigma",
		"unet_config": {
			"input_size"  : 2048//8,
			"token_num"   :  300,
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1152,
			"micro_condition": False,
			"pe_interpolation": 4,
			"model_max_length": 300,
		},
		"sampling_settings" : sampling_settings,
	},
	"PixArt_XL_2": { # models/PixArt
		"target": "PixArt",
		"unet_config": {
			"input_size"  :  512//8,
			"token_num"   :  120,
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1152,
			"pe_interpolation": 1,
		},
		"sampling_settings" : sampling_settings,
	},
}

pixart_conf.update({ # controlnet models
	"ControlPixArtHalf": {
		"target": "ControlPixArtHalf",
		"unet_config": pixart_conf["PixArt_XL_2"]["unet_config"],
		"sampling_settings": pixart_conf["PixArt_XL_2"]["sampling_settings"],
	},
	"ControlPixArtMSHalf": {
		"target": "ControlPixArtMSHalf",
		"unet_config": pixart_conf["PixArtMS_XL_2"]["unet_config"],
		"sampling_settings": pixart_conf["PixArtMS_XL_2"]["sampling_settings"],
	}
})

pixart_res = {
	"PixArtMS_XL_2": { # models/PixArtMS 1024x1024
		'0.25': [512, 2048], '0.26': [512, 1984], '0.27': [512, 1920], '0.28': [512, 1856],
		'0.32': [576, 1792], '0.33': [576, 1728], '0.35': [576, 1664], '0.40': [640, 1600],
		'0.42': [640, 1536], '0.48': [704, 1472], '0.50': [704, 1408], '0.52': [704, 1344],
		'0.57': [768, 1344], '0.60': [768, 1280], '0.68': [832, 1216], '0.72': [832, 1152],
		'0.78': [896, 1152], '0.82': [896, 1088], '0.88': [960, 1088], '0.94': [960, 1024],
		'1.00': [1024,1024], '1.07': [1024, 960], '1.13': [1088, 960], '1.21': [1088, 896],
		'1.29': [1152, 896], '1.38': [1152, 832], '1.46': [1216, 832], '1.67': [1280, 768],
		'1.75': [1344, 768], '2.00': [1408, 704], '2.09': [1472, 704], '2.40': [1536, 640],
		'2.50': [1600, 640], '2.89': [1664, 576], '3.00': [1728, 576], '3.11': [1792, 576],
		'3.62': [1856, 512], '3.75': [1920, 512], '3.88': [1984, 512], '4.00': [2048, 512],
	},
	"PixArt_XL_2": { # models/PixArt 512x512
		'0.25': [256,1024], '0.26': [256, 992], '0.27': [256, 960], '0.28': [256, 928],
		'0.32': [288, 896], '0.33': [288, 864], '0.35': [288, 832], '0.40': [320, 800],
		'0.42': [320, 768], '0.48': [352, 736], '0.50': [352, 704], '0.52': [352, 672],
		'0.57': [384, 672], '0.60': [384, 640], '0.68': [416, 608], '0.72': [416, 576],
		'0.78': [448, 576], '0.82': [448, 544], '0.88': [480, 544], '0.94': [480, 512],
		'1.00': [512, 512], '1.07': [512, 480], '1.13': [544, 480], '1.21': [544, 448],
		'1.29': [576, 448], '1.38': [576, 416], '1.46': [608, 416], '1.67': [640, 384],
		'1.75': [672, 384], '2.00': [704, 352], '2.09': [736, 352], '2.40': [768, 320],
		'2.50': [800, 320], '2.89': [832, 288], '3.00': [864, 288], '3.11': [896, 288],
		'3.62': [928, 256], '3.75': [960, 256], '3.88': [992, 256], '4.00': [1024,256]
	},
	"PixArtMS_Sigma_XL_2_2K": {
		'0.25': [1024, 4096], '0.26': [1024, 3968], '0.27': [1024, 3840], '0.28': [1024, 3712],
		'0.32': [1152, 3584], '0.33': [1152, 3456], '0.35': [1152, 3328], '0.40': [1280, 3200],
		'0.42': [1280, 3072], '0.48': [1408, 2944], '0.50': [1408, 2816], '0.52': [1408, 2688],
		'0.57': [1536, 2688], '0.60': [1536, 2560], '0.68': [1664, 2432], '0.72': [1664, 2304],
		'0.78': [1792, 2304], '0.82': [1792, 2176], '0.88': [1920, 2176], '0.94': [1920, 2048],
		'1.00': [2048, 2048], '1.07': [2048, 1920], '1.13': [2176, 1920], '1.21': [2176, 1792],
		'1.29': [2304, 1792], '1.38': [2304, 1664], '1.46': [2432, 1664], '1.67': [2560, 1536],
		'1.75': [2688, 1536], '2.00': [2816, 1408], '2.09': [2944, 1408], '2.40': [3072, 1280],
		'2.50': [3200, 1280], '2.89': [3328, 1152], '3.00': [3456, 1152], '3.11': [3584, 1152],
		'3.62': [3712, 1024], '3.75': [3840, 1024], '3.88': [3968, 1024], '4.00': [4096, 1024]
	}
}
# These should be the same
pixart_res.update({
	"PixArtMS_Sigma_XL_2": pixart_res["PixArtMS_XL_2"],
	"PixArtMS_Sigma_XL_2_512": pixart_res["PixArt_XL_2"],
})