import comfy.utils
import logging

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora, to_load):
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None
        if dora_scale_name in lora.keys():
            dora_scale = lora[dora_scale_name]
            loaded_keys.add(dora_scale_name)

        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        diffusers2_lora = "{}.lora_B.weight".format(x)
        diffusers3_lora = "{}.lora.up.weight".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif diffusers2_lora in lora.keys():
            A_name = diffusers2_lora
            B_name = "{}.lora_A.weight".format(x)
            mid_name = None
        elif diffusers3_lora in lora.keys():
            A_name = diffusers3_lora
            B_name = "{}.lora.down.weight".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name ="{}.lora_linear_layer.down.weight".format(x)
            mid_name = None

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = ("lora", (lora[A_name], lora[B_name], alpha, mid, dora_scale))
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)


        ######## loha
        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            patch_dict[to_load[x]] = ("loha", (lora[hada_w1_a_name], lora[hada_w1_b_name], alpha, lora[hada_w2_a_name], lora[hada_w2_b_name], hada_t1, hada_t2, dora_scale))
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)


        ######## lokr
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (lokr_w1 is not None) or (lokr_w2 is not None) or (lokr_w1_a is not None) or (lokr_w2_a is not None):
            patch_dict[to_load[x]] = ("lokr", (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2, dora_scale))

        #glora
        a1_name = "{}.a1.weight".format(x)
        a2_name = "{}.a2.weight".format(x)
        b1_name = "{}.b1.weight".format(x)
        b2_name = "{}.b2.weight".format(x)
        if a1_name in lora:
            patch_dict[to_load[x]] = ("glora", (lora[a1_name], lora[a2_name], lora[b1_name], lora[b2_name], alpha, dora_scale))
            loaded_keys.add(a1_name)
            loaded_keys.add(a2_name)
            loaded_keys.add(b1_name)
            loaded_keys.add(b2_name)

        w_norm_name = "{}.w_norm".format(x)
        b_norm_name = "{}.b_norm".format(x)
        w_norm = lora.get(w_norm_name, None)
        b_norm = lora.get(b_norm_name, None)

        if w_norm is not None:
            loaded_keys.add(w_norm_name)
            patch_dict[to_load[x]] = ("diff", (w_norm,))
            if b_norm is not None:
                loaded_keys.add(b_norm_name)
                patch_dict["{}.bias".format(to_load[x][:-len(".weight")])] = ("diff", (b_norm,))

        diff_name = "{}.diff".format(x)
        diff_weight = lora.get(diff_name, None)
        if diff_weight is not None:
            patch_dict[to_load[x]] = ("diff", (diff_weight,))
            loaded_keys.add(diff_name)

        diff_bias_name = "{}.diff_b".format(x)
        diff_bias = lora.get(diff_bias_name, None)
        if diff_bias is not None:
            patch_dict["{}.bias".format(to_load[x][:-len(".weight")])] = ("diff", (diff_bias,))
            loaded_keys.add(diff_bias_name)

    for x in lora.keys():
        if x not in loaded_keys:
            logging.warning("lora key not loaded: {}".format(x))

    return patch_dict

def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    for b in range(32): #TODO: clean up
        for c in LORA_CLIP_MAP:
            k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                if clip_l_present:
                    lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                    key_map[lora_key] = k
                    lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                else:
                    lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #TODO: test if this is correct for SDXL-Refiner
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                    lora_key = "lora_prior_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #cascade lora: TODO put lora key prefix in the model config
                    key_map[lora_key] = k

    for k in sdk: #OneTrainer SD3 lora
        if k.startswith("t5xxl.transformer.") and k.endswith(".weight"):
            l_key = k[len("t5xxl.transformer."):-len(".weight")]
            lora_key = "lora_te3_{}".format(l_key.replace(".", "_"))
            key_map[lora_key] = k

    k = "clip_g.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_prior_te_text_projection"] = k #cascade lora?
        # key_map["text_encoder.text_projection"] = k #TODO: check if other lora have the text_projection too
        key_map["lora_te2_text_projection"] = k #OneTrainer SD3 lora

    k = "clip_l.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_te1_text_projection"] = k #OneTrainer SD3 lora, not necessary but omits warning

    return key_map

def model_lora_keys_unet(model, key_map={}):
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["lora_prior_unet_{}".format(key_lora)] = k #cascade lora: TODO put lora key prefix in the model config

    diffusers_keys = comfy.utils.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    if isinstance(model, comfy.model_base.SD3): #Diffusers lora SD3
        diffusers_keys = comfy.utils.mmdit_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #regular diffusers sd3 lora format
                key_map[key_lora] = to

                key_lora = "base_model.model.{}".format(k[:-len(".weight")]) #format for flash-sd3 lora and others?
                key_map[key_lora] = to

                key_lora = "lora_transformer_{}".format(k[:-len(".weight")].replace(".", "_")) #OneTrainer lora
                key_map[key_lora] = to

    if isinstance(model, comfy.model_base.AuraFlow): #Diffusers lora AuraFlow
        diffusers_keys = comfy.utils.auraflow_to_diffusers(model.model_config.unet_config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_lora = "transformer.{}".format(k[:-len(".weight")]) #simpletrainer and probably regular diffusers lora format
                key_map[key_lora] = to

    return key_map
