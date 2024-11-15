import comfy_extras.nodes_model_merging

class ModelMergeSD1(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(12):
            arg_dict["input_blocks.{}.".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}.".format(i)] = argument

        for i in range(12):
            arg_dict["output_blocks.{}.".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}


class ModelMergeSDXL(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(9):
            arg_dict["input_blocks.{}".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}".format(i)] = argument

        for i in range(9):
            arg_dict["output_blocks.{}".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}

class ModelMergeSD3_2B(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embed."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["context_embedder."] = argument
        arg_dict["y_embedder."] = argument
        arg_dict["t_embedder."] = argument

        for i in range(24):
            arg_dict["joint_blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}


class ModelMergeAuraflow(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["init_x_linear."] = argument
        arg_dict["positional_encoding"] = argument
        arg_dict["cond_seq_linear."] = argument
        arg_dict["register_tokens"] = argument
        arg_dict["t_embedder."] = argument

        for i in range(4):
            arg_dict["double_layers.{}.".format(i)] = argument

        for i in range(32):
            arg_dict["single_layers.{}.".format(i)] = argument

        arg_dict["modF."] = argument
        arg_dict["final_linear."] = argument

        return {"required": arg_dict}

class ModelMergeFlux1(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["img_in."] = argument
        arg_dict["time_in."] = argument
        arg_dict["guidance_in"] = argument
        arg_dict["vector_in."] = argument
        arg_dict["txt_in."] = argument

        for i in range(19):
            arg_dict["double_blocks.{}.".format(i)] = argument

        for i in range(38):
            arg_dict["single_blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeSD35_Large(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embed."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["context_embedder."] = argument
        arg_dict["y_embedder."] = argument
        arg_dict["t_embedder."] = argument

        for i in range(38):
            arg_dict["joint_blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

class ModelMergeMochiPreview(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_frequencies."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["t5_y_embedder."] = argument
        arg_dict["t5_yproj."] = argument

        for i in range(48):
            arg_dict["blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

NODE_CLASS_MAPPINGS = {
    "ModelMergeSD1": ModelMergeSD1,
    "ModelMergeSD2": ModelMergeSD1, #SD1 and SD2 have the same blocks
    "ModelMergeSDXL": ModelMergeSDXL,
    "ModelMergeSD3_2B": ModelMergeSD3_2B,
    "ModelMergeAuraflow": ModelMergeAuraflow,
    "ModelMergeFlux1": ModelMergeFlux1,
    "ModelMergeSD35_Large": ModelMergeSD35_Large,
    "ModelMergeMochiPreview": ModelMergeMochiPreview,
}
