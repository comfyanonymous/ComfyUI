"""
List of all DiT model types / settings
"""
sampling_settings = {
	"beta_schedule" : "sqrt_linear",
	"linear_start"  : 0.0001,
	"linear_end"    : 0.02,
	"timesteps"     : 1000,
}

dit_conf = {
	"XL/2": { # DiT_XL_2
		"unet_config": {
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1152,
		},
		"sampling_settings" : sampling_settings,
	},
	"XL/4": { # DiT_XL_4
		"unet_config": {
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    4,
			"hidden_size" : 1152,
		},
		"sampling_settings" : sampling_settings,
	},
	"XL/8": { # DiT_XL_8
		"unet_config": {
			"depth"       :   28,
			"num_heads"   :   16,
			"patch_size"  :    8,
			"hidden_size" : 1152,
		},
		"sampling_settings" : sampling_settings,
	},
	"L/2": { # DiT_L_2
		"unet_config": {
			"depth"       :   24,
			"num_heads"   :   16,
			"patch_size"  :    2,
			"hidden_size" : 1024,
		},
		"sampling_settings" : sampling_settings,
	},
	"L/4": { # DiT_L_4
		"unet_config": {
			"depth"       :   24,
			"num_heads"   :   16,
			"patch_size"  :    4,
			"hidden_size" : 1024,
		},
		"sampling_settings" : sampling_settings,
	},
	"L/8": { # DiT_L_8
		"unet_config": {
			"depth"       :   24,
			"num_heads"   :   16,
			"patch_size"  :    8,
			"hidden_size" : 1024,
		},
		"sampling_settings" : sampling_settings,
	},
	"B/2": { # DiT_B_2
		"unet_config": {
			"depth"       :   12,
			"num_heads"   :   12,
			"patch_size"  :    2,
			"hidden_size" :  768,
		},
		"sampling_settings" : sampling_settings,
	},
	"B/4": { # DiT_B_4
		"unet_config": {
			"depth"       :   12,
			"num_heads"   :   12,
			"patch_size"  :    4,
			"hidden_size" :  768,
		},
		"sampling_settings" : sampling_settings,
	},
	"B/8": { # DiT_B_8
		"unet_config": {
			"depth"       :   12,
			"num_heads"   :   12,
			"patch_size"  :    8,
			"hidden_size" :  768,
		},
		"sampling_settings" : sampling_settings,
	},
	"S/2": { # DiT_S_2
		"unet_config": {
			"depth"       :   12,
			"num_heads"   :    6,
			"patch_size"  :    2,
			"hidden_size" :  384,
		},
		"sampling_settings" : sampling_settings,
	},
	"S/4": { # DiT_S_4
		"unet_config": {
			"depth"       :   12,
			"num_heads"   :    6,
			"patch_size"  :    4,
			"hidden_size" :  384,
		},
		"sampling_settings" : sampling_settings,
	},
	"S/8": { # DiT_S_8
		"unet_config": {
			"depth"       :   12,
			"num_heads"   :    6,
			"patch_size"  :    8,
			"hidden_size" :  384,
		},
		"sampling_settings" : sampling_settings,
	},
}