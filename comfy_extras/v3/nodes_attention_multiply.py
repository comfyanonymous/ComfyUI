from comfy_api.latest import io


def attention_multiply(attn, model, q, k, v, out):
    m = model.clone()
    sd = model.model_state_dict()

    for key in sd:
        if key.endswith("{}.to_q.bias".format(attn)) or key.endswith("{}.to_q.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, q)
        if key.endswith("{}.to_k.bias".format(attn)) or key.endswith("{}.to_k.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, k)
        if key.endswith("{}.to_v.bias".format(attn)) or key.endswith("{}.to_v.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, v)
        if key.endswith("{}.to_out.0.bias".format(attn)) or key.endswith("{}.to_out.0.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, out)
    return m


class UNetSelfAttentionMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UNetSelfAttentionMultiply_V3",
            category="_for_testing/attention_experiments",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("q", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("k", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("v", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("out", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, q, k, v, out) -> io.NodeOutput:
        return io.NodeOutput(attention_multiply("attn1", model, q, k, v, out))


class UNetCrossAttentionMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UNetCrossAttentionMultiply_V3",
            category="_for_testing/attention_experiments",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("q", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("k", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("v", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("out", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, q, k, v, out) -> io.NodeOutput:
        return io.NodeOutput(attention_multiply("attn2", model, q, k, v, out))


class CLIPAttentionMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPAttentionMultiply_V3",
            category="_for_testing/attention_experiments",
            inputs=[
                io.Clip.Input("clip"),
                io.Float.Input("q", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("k", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("v", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("out", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[io.Clip.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, q, k, v, out) -> io.NodeOutput:
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
        return io.NodeOutput(m)


class UNetTemporalAttentionMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UNetTemporalAttentionMultiply_V3",
            category="_for_testing/attention_experiments",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("self_structural", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("self_temporal", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("cross_structural", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("cross_temporal", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, self_structural, self_temporal, cross_structural, cross_temporal) -> io.NodeOutput:
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
        return io.NodeOutput(m)


NODES_LIST: list[type[io.ComfyNode]] = [
    CLIPAttentionMultiply,
    UNetCrossAttentionMultiply,
    UNetSelfAttentionMultiply,
    UNetTemporalAttentionMultiply,
]
