import comfy_extras.nodes_model_merging

from comfy_api.latest import io, ComfyExtension
from typing_extensions import override


class ModelMergeSD1(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("time_embed.", **argument))
        inputs.append(io.Float.Input("label_emb.", **argument))

        for i in range(12):
            inputs.append(io.Float.Input("input_blocks.{}.".format(i), **argument))

        for i in range(3):
            inputs.append(io.Float.Input("middle_block.{}.".format(i), **argument))

        for i in range(12):
            inputs.append(io.Float.Input("output_blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("out.", **argument))

        return io.Schema(
            node_id="ModelMergeSD1",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeSD2(ModelMergeSD1):
    # SD1 and SD2 have the same blocks
    @classmethod
    def define_schema(cls):
        schema = ModelMergeSD1.define_schema()
        schema.node_id = "ModelMergeSD2"
        return schema


class ModelMergeSDXL(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("time_embed.", **argument))
        inputs.append(io.Float.Input("label_emb.", **argument))

        for i in range(9):
            inputs.append(io.Float.Input("input_blocks.{}".format(i), **argument))

        for i in range(3):
            inputs.append(io.Float.Input("middle_block.{}".format(i), **argument))

        for i in range(9):
            inputs.append(io.Float.Input("output_blocks.{}".format(i), **argument))

        inputs.append(io.Float.Input("out.", **argument))

        return io.Schema(
            node_id="ModelMergeSDXL",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeSD3_2B(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_embed.", **argument))
        inputs.append(io.Float.Input("x_embedder.", **argument))
        inputs.append(io.Float.Input("context_embedder.", **argument))
        inputs.append(io.Float.Input("y_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))

        for i in range(24):
            inputs.append(io.Float.Input("joint_blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeSD3_2B",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeAuraflow(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("init_x_linear.", **argument))
        inputs.append(io.Float.Input("positional_encoding", **argument))
        inputs.append(io.Float.Input("cond_seq_linear.", **argument))
        inputs.append(io.Float.Input("register_tokens", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))

        for i in range(4):
            inputs.append(io.Float.Input("double_layers.{}.".format(i), **argument))

        for i in range(32):
            inputs.append(io.Float.Input("single_layers.{}.".format(i), **argument))

        inputs.append(io.Float.Input("modF.", **argument))
        inputs.append(io.Float.Input("final_linear.", **argument))

        return io.Schema(
            node_id="ModelMergeAuraflow",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeFlux1(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("img_in.", **argument))
        inputs.append(io.Float.Input("time_in.", **argument))
        inputs.append(io.Float.Input("guidance_in", **argument))
        inputs.append(io.Float.Input("vector_in.", **argument))
        inputs.append(io.Float.Input("txt_in.", **argument))

        for i in range(19):
            inputs.append(io.Float.Input("double_blocks.{}.".format(i), **argument))

        for i in range(38):
            inputs.append(io.Float.Input("single_blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeFlux1",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeSD35_Large(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_embed.", **argument))
        inputs.append(io.Float.Input("x_embedder.", **argument))
        inputs.append(io.Float.Input("context_embedder.", **argument))
        inputs.append(io.Float.Input("y_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))

        for i in range(38):
            inputs.append(io.Float.Input("joint_blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeSD35_Large",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeMochiPreview(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_frequencies.", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))
        inputs.append(io.Float.Input("t5_y_embedder.", **argument))
        inputs.append(io.Float.Input("t5_yproj.", **argument))

        for i in range(48):
            inputs.append(io.Float.Input("blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeMochiPreview",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeLTXV(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("patchify_proj.", **argument))
        inputs.append(io.Float.Input("adaln_single.", **argument))
        inputs.append(io.Float.Input("caption_projection.", **argument))

        for i in range(28):
            inputs.append(io.Float.Input("transformer_blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("scale_shift_table", **argument))
        inputs.append(io.Float.Input("proj_out.", **argument))

        return io.Schema(
            node_id="ModelMergeLTXV",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeCosmos7B(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_embedder.", **argument))
        inputs.append(io.Float.Input("extra_pos_embedder.", **argument))
        inputs.append(io.Float.Input("x_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))
        inputs.append(io.Float.Input("affline_norm.", **argument))

        for i in range(28):
            inputs.append(io.Float.Input("blocks.block{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeCosmos7B",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeCosmos14B(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_embedder.", **argument))
        inputs.append(io.Float.Input("extra_pos_embedder.", **argument))
        inputs.append(io.Float.Input("x_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))
        inputs.append(io.Float.Input("affline_norm.", **argument))

        for i in range(36):
            inputs.append(io.Float.Input("blocks.block{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeCosmos14B",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeWAN2_1(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("patch_embedding.", **argument))
        inputs.append(io.Float.Input("time_embedding.", **argument))
        inputs.append(io.Float.Input("time_projection.", **argument))
        inputs.append(io.Float.Input("text_embedding.", **argument))
        inputs.append(io.Float.Input("img_emb.", **argument))

        for i in range(40):
            inputs.append(io.Float.Input("blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("head.", **argument))

        return io.Schema(
            node_id="ModelMergeWAN2_1",
            category="advanced/model_merging/model_specific",
            description="1.3B model has 30 blocks, 14B model has 40 blocks. Image to video model has the extra img_emb.",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeCosmosPredict2_2B(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_embedder.", **argument))
        inputs.append(io.Float.Input("x_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedding_norm.", **argument))

        for i in range(28):
            inputs.append(io.Float.Input("blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeCosmosPredict2_2B",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeCosmosPredict2_14B(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_embedder.", **argument))
        inputs.append(io.Float.Input("x_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedder.", **argument))
        inputs.append(io.Float.Input("t_embedding_norm.", **argument))

        for i in range(36):
            inputs.append(io.Float.Input("blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("final_layer.", **argument))

        return io.Schema(
            node_id="ModelMergeCosmosPredict2_14B",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergeQwenImage(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
        ]

        argument = dict(default=1.0, min=0.0, max=1.0, step=0.01)

        inputs.append(io.Float.Input("pos_embeds.", **argument))
        inputs.append(io.Float.Input("img_in.", **argument))
        inputs.append(io.Float.Input("txt_norm.", **argument))
        inputs.append(io.Float.Input("txt_in.", **argument))
        inputs.append(io.Float.Input("time_text_embed.", **argument))

        for i in range(60):
            inputs.append(io.Float.Input("transformer_blocks.{}.".format(i), **argument))

        inputs.append(io.Float.Input("proj_out.", **argument))

        return io.Schema(
            node_id="ModelMergeQwenImage",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[io.Model.Output()],
        )


class ModelMergingModelSpecificExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ModelMergeSD1,
            ModelMergeSD2,
            ModelMergeSDXL,
            ModelMergeSD3_2B,
            ModelMergeAuraflow,
            ModelMergeFlux1,
            ModelMergeSD35_Large,
            ModelMergeMochiPreview,
            ModelMergeLTXV,
            ModelMergeCosmos7B,
            ModelMergeCosmos14B,
            ModelMergeWAN2_1,
            ModelMergeCosmosPredict2_2B,
            ModelMergeCosmosPredict2_14B,
            ModelMergeQwenImage,
        ]


async def comfy_entrypoint() -> ModelMergingModelSpecificExtension:
    return ModelMergingModelSpecificExtension()
