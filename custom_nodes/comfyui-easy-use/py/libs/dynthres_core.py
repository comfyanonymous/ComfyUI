import torch, math

######################### DynThresh Core #########################

class DynThresh:

    Modes = ["Constant", "Linear Down", "Cosine Down", "Half Cosine Down", "Linear Up", "Cosine Up", "Half Cosine Up", "Power Up", "Power Down", "Linear Repeating", "Cosine Repeating", "Sawtooth"]
    Startpoints = ["MEAN", "ZERO"]
    Variabilities = ["AD", "STD"]

    def __init__(self, mimic_scale, threshold_percentile, mimic_mode, mimic_scale_min, cfg_mode, cfg_scale_min, sched_val, experiment_mode, max_steps, separate_feature_channels, scaling_startpoint, variability_measure, interpolate_phi):
        self.mimic_scale = mimic_scale
        self.threshold_percentile = threshold_percentile
        self.mimic_mode = mimic_mode
        self.cfg_mode = cfg_mode
        self.max_steps = max_steps
        self.cfg_scale_min = cfg_scale_min
        self.mimic_scale_min = mimic_scale_min
        self.experiment_mode = experiment_mode
        self.sched_val = sched_val
        self.sep_feat_channels = separate_feature_channels
        self.scaling_startpoint = scaling_startpoint
        self.variability_measure = variability_measure
        self.interpolate_phi = interpolate_phi

    def interpret_scale(self, scale, mode, min):
        scale -= min
        max = self.max_steps - 1
        frac = self.step / max
        if mode == "Constant":
            pass
        elif mode == "Linear Down":
            scale *= 1.0 - frac
        elif mode == "Half Cosine Down":
            scale *= math.cos(frac)
        elif mode == "Cosine Down":
            scale *= math.cos(frac * 1.5707)
        elif mode == "Linear Up":
            scale *= frac
        elif mode == "Half Cosine Up":
            scale *= 1.0 - math.cos(frac)
        elif mode == "Cosine Up":
            scale *= 1.0 - math.cos(frac * 1.5707)
        elif mode == "Power Up":
            scale *= math.pow(frac, self.sched_val)
        elif mode == "Power Down":
            scale *= 1.0 - math.pow(frac, self.sched_val)
        elif mode == "Linear Repeating":
            portion = (frac * self.sched_val) % 1.0
            scale *= (0.5 - portion) * 2 if portion < 0.5 else (portion - 0.5) * 2
        elif mode == "Cosine Repeating":
            scale *= math.cos(frac * 6.28318 * self.sched_val) * 0.5 + 0.5
        elif mode == "Sawtooth":
            scale *= (frac * self.sched_val) % 1.0
        scale += min
        return scale

    def dynthresh(self, cond, uncond, cfg_scale, weights):
        mimic_scale = self.interpret_scale(self.mimic_scale, self.mimic_mode, self.mimic_scale_min)
        cfg_scale = self.interpret_scale(cfg_scale, self.cfg_mode, self.cfg_scale_min)
        # uncond shape is (batch, 4, height, width)
        conds_per_batch = cond.shape[0] / uncond.shape[0]
        assert conds_per_batch == int(conds_per_batch), "Expected # of conds per batch to be constant across batches"
        cond_stacked = cond.reshape((-1, int(conds_per_batch)) + uncond.shape[1:])

        ### Normal first part of the CFG Scale logic, basically
        diff = cond_stacked - uncond.unsqueeze(1)
        if weights is not None:
            diff = diff * weights
        relative = diff.sum(1)

        ### Get the normal result for both mimic and normal scale
        mim_target = uncond + relative * mimic_scale
        cfg_target = uncond + relative * cfg_scale
        ### If we weren't doing mimic scale, we'd just return cfg_target here

        ### Now recenter the values relative to their average rather than absolute, to allow scaling from average
        mim_flattened = mim_target.flatten(2)
        cfg_flattened = cfg_target.flatten(2)
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        mim_centered = mim_flattened - mim_means
        cfg_centered = cfg_flattened - cfg_means

        if self.sep_feat_channels:
            if self.variability_measure == 'STD':
                mim_scaleref = mim_centered.std(dim=2).unsqueeze(2)
                cfg_scaleref = cfg_centered.std(dim=2).unsqueeze(2)
            else: # 'AD'
                mim_scaleref = mim_centered.abs().max(dim=2).values.unsqueeze(2)
                cfg_scaleref = torch.quantile(cfg_centered.abs(), self.threshold_percentile, dim=2).unsqueeze(2)

        else:
            if self.variability_measure == 'STD':
                mim_scaleref = mim_centered.std()
                cfg_scaleref = cfg_centered.std()
            else: # 'AD'
                mim_scaleref = mim_centered.abs().max()
                cfg_scaleref = torch.quantile(cfg_centered.abs(), self.threshold_percentile)

        if self.scaling_startpoint == 'ZERO':
            scaling_factor = mim_scaleref / cfg_scaleref
            result = cfg_flattened * scaling_factor

        else: # 'MEAN'
            if self.variability_measure == 'STD':
                cfg_renormalized = (cfg_centered / cfg_scaleref) * mim_scaleref
            else: # 'AD'
                ### Get the maximum value of all datapoints (with an optional threshold percentile on the uncond)
                max_scaleref = torch.maximum(mim_scaleref, cfg_scaleref)
                ### Clamp to the max
                cfg_clamped = cfg_centered.clamp(-max_scaleref, max_scaleref)
                ### Now shrink from the max to normalize and grow to the mimic scale (instead of the CFG scale)
                cfg_renormalized = (cfg_clamped / max_scaleref) * mim_scaleref

            ### Now add it back onto the averages to get into real scale again and return
            result = cfg_renormalized + cfg_means

        actual_res = result.unflatten(2, mim_target.shape[2:])

        if self.interpolate_phi != 1.0:
            actual_res = actual_res * self.interpolate_phi + cfg_target * (1.0 - self.interpolate_phi)

        if self.experiment_mode == 1:
            num = actual_res.cpu().numpy()
            for y in range(0, 64):
                for x in range (0, 64):
                    if num[0][0][y][x] > 1.0:
                        num[0][1][y][x] *= 0.5
                    if num[0][1][y][x] > 1.0:
                        num[0][1][y][x] *= 0.5
                    if num[0][2][y][x] > 1.5:
                        num[0][2][y][x] *= 0.5
            actual_res = torch.from_numpy(num).to(device=uncond.device)
        elif self.experiment_mode == 2:
            num = actual_res.cpu().numpy()
            for y in range(0, 64):
                for x in range (0, 64):
                    over_scale = False
                    for z in range(0, 4):
                        if abs(num[0][z][y][x]) > 1.5:
                            over_scale = True
                    if over_scale:
                        for z in range(0, 4):
                            num[0][z][y][x] *= 0.7
            actual_res = torch.from_numpy(num).to(device=uncond.device)
        elif self.experiment_mode == 3:
            coefs = torch.tensor([
                #  R       G        B      W
                [0.298,   0.207,  0.208, 0.0], # L1
                [0.187,   0.286,  0.173, 0.0], # L2
                [-0.158,  0.189,  0.264, 0.0], # L3
                [-0.184, -0.271, -0.473, 1.0], # L4
            ], device=uncond.device)
            res_rgb = torch.einsum("laxy,ab -> lbxy", actual_res, coefs)
            max_r, max_g, max_b, max_w = res_rgb[0][0].max(), res_rgb[0][1].max(), res_rgb[0][2].max(), res_rgb[0][3].max()
            max_rgb = max(max_r, max_g, max_b)
            print(f"test max = r={max_r}, g={max_g}, b={max_b}, w={max_w}, rgb={max_rgb}")
            if self.step / (self.max_steps - 1) > 0.2:
                if max_rgb < 2.0 and max_w < 3.0:
                    res_rgb /= max_rgb / 2.4
            else:
                if max_rgb > 2.4 and max_w > 3.0:
                    res_rgb /= max_rgb / 2.4
            actual_res = torch.einsum("laxy,ab -> lbxy", res_rgb, coefs.inverse())

        return actual_res