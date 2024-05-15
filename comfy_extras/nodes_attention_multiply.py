
def attention_multiply(attn, model, q, k, v, out, block, layer):
    m = model.clone()
    sd = model.model_state_dict()

    for key in sd:
        if f"{block}_blocks.{layer}" not in key and block != "all":
            continue
        if key.endswith("{}.to_q.bias".format(attn)) or key.endswith("{}.to_q.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, q)
        if key.endswith("{}.to_k.bias".format(attn)) or key.endswith("{}.to_k.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, k)
        if key.endswith("{}.to_v.bias".format(attn)) or key.endswith("{}.to_v.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, v)
        if key.endswith("{}.to_out.0.bias".format(attn)) or key.endswith("{}.to_out.0.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, out)

    return m


class UNetSelfAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "q": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "v": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "block_id_input":  ("STRING", {"multiline": False}, {"default": ""}),
                              "block_id_middle": ("STRING", {"multiline": False}, {"default": ""}),
                              "block_id_output": ("STRING", {"multiline": False}, {"default": ""}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, model, q, k, v, out, block_id_input, block_id_middle, block_id_output):
        block_layers = {"input": block_id_input, "middle": block_id_middle, "output": block_id_output}
        m = model.clone()
        if all(value == "" for value in block_layers.values()):
            m = attention_multiply("attn1", m, q, k, v, out, "all", "all")
        else:
            for block in block_layers:
                for block_id in block_layers[block].split(","):
                    if block_id != "":
                        block_id = int(block_id)
                        m = attention_multiply("attn1", m, q, k, v, out, block, block_id)
        return (m, )

class UNetCrossAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "q": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "v": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "block_id_input":  ("STRING", {"multiline": False}, {"default": ""}),
                              "block_id_middle": ("STRING", {"multiline": False}, {"default": ""}),
                              "block_id_output": ("STRING", {"multiline": False}, {"default": ""}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, model, q, k, v, out, block_id_input, block_id_middle, block_id_output):
        block_layers = {"input": block_id_input, "middle": block_id_middle, "output": block_id_output}
        m = model.clone()
        if all(value == "" for value in block_layers.values()):
            m = attention_multiply("attn2", m, q, k, v, out, "all", "all")
        else:
            for block in block_layers:
                for block_id in block_layers[block].split(","):
                    if block_id != "":
                        block_id = int(block_id)
                        m = attention_multiply("attn2", m, q, k, v, out, block, block_id)
        return (m, )

class CLIPAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip": ("CLIP",),
                              "q": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "k": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "v": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, clip, q, k, v, out):
        m = clip.clone()
        sd = m.patcher.model_state_dict()

        for key in sd:
            if key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.q_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, q)
            if key.endswith("self_attn.k_proj.weight") or key.endswith("self_attn.k_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, k)
            if key.endswith("self_attn.v_proj.weight") or key.endswith("self_attn.v_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, v)
            if key.endswith("self_attn.out_proj.weight") or key.endswith("self_attn.out_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, out)
        return (m, )

class UNetTemporalAttentionMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "self_structural": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "self_temporal": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "cross_structural": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "cross_temporal": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing/attention_experiments"

    def patch(self, model, self_structural, self_temporal, cross_structural, cross_temporal):
        m = model.clone()
        sd = model.model_state_dict()

        for k in sd:
            if (k.endswith("attn1.to_out.0.bias") or k.endswith("attn1.to_out.0.weight")):
                if '.time_stack.' in k:
                    m.add_patches({k: (None,)}, 0.0, self_temporal)
                else:
                    m.add_patches({k: (None,)}, 0.0, self_structural)
            elif (k.endswith("attn2.to_out.0.bias") or k.endswith("attn2.to_out.0.weight")):
                if '.time_stack.' in k:
                    m.add_patches({k: (None,)}, 0.0, cross_temporal)
                else:
                    m.add_patches({k: (None,)}, 0.0, cross_structural)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "UNetSelfAttentionMultiply": UNetSelfAttentionMultiply,
    "UNetCrossAttentionMultiply": UNetCrossAttentionMultiply,
    "CLIPAttentionMultiply": CLIPAttentionMultiply,
    "UNetTemporalAttentionMultiply": UNetTemporalAttentionMultiply,
}
