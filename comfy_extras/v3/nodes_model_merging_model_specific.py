from __future__ import annotations

from comfy_api.latest import io
from comfy_extras.v3.nodes_model_merging import ModelMergeBlocks


class ModelMergeSD1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("time_embed.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("label_emb.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(12):
            inputs.append(io.Float.Input(f"input_blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        for i in range(3):
            inputs.append(io.Float.Input(f"middle_block.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        for i in range(12):
            inputs.append(io.Float.Input(f"output_blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("out.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeSD1_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeSDXL(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("time_embed.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("label_emb.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(9):
            inputs.append(io.Float.Input(f"input_blocks.{i}", default=1.0, min=0.0, max=1.0, step=0.01))

        for i in range(3):
            inputs.append(io.Float.Input(f"middle_block.{i}", default=1.0, min=0.0, max=1.0, step=0.01))

        for i in range(9):
            inputs.append(io.Float.Input(f"output_blocks.{i}", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("out.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeSDXL_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeSD3_2B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("pos_embed.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("x_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("context_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("y_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(24):
            inputs.append(io.Float.Input(f"joint_blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeSD3_2B_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeAuraflow(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("init_x_linear.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("positional_encoding", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("cond_seq_linear.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("register_tokens", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(4):
            inputs.append(io.Float.Input(f"double_layers.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        for i in range(32):
            inputs.append(io.Float.Input(f"single_layers.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.extend([
            io.Float.Input("modF.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("final_linear.", default=1.0, min=0.0, max=1.0, step=0.01)
        ])

        return io.Schema(
            node_id="ModelMergeAuraflow_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeFlux1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("img_in.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("time_in.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("guidance_in", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("vector_in.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("txt_in.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(19):
            inputs.append(io.Float.Input(f"double_blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        for i in range(38):
            inputs.append(io.Float.Input(f"single_blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeFlux1_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeSD35_Large(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("pos_embed.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("x_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("context_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("y_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(38):
            inputs.append(io.Float.Input(f"joint_blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeSD35_Large_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeMochiPreview(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("pos_frequencies.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t5_y_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t5_yproj.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(48):
            inputs.append(io.Float.Input(f"blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeMochiPreview_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeLTXV(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("patchify_proj.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("adaln_single.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("caption_projection.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(28):
            inputs.append(io.Float.Input(f"transformer_blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.extend([
            io.Float.Input("scale_shift_table", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("proj_out.", default=1.0, min=0.0, max=1.0, step=0.01)
        ])

        return io.Schema(
            node_id="ModelMergeLTXV_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeCosmos7B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("pos_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("extra_pos_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("x_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("affline_norm.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(28):
            inputs.append(io.Float.Input(f"blocks.block{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeCosmos7B_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeCosmos14B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("pos_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("extra_pos_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("x_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("affline_norm.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(36):
            inputs.append(io.Float.Input(f"blocks.block{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeCosmos14B_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeWAN2_1(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("patch_embedding.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("time_embedding.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("time_projection.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("text_embedding.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("img_emb.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(40):
            inputs.append(io.Float.Input(f"blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("head.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeWAN2_1_V3",
            category="advanced/model_merging/model_specific",
            description="1.3B model has 30 blocks, 14B model has 40 blocks. Image to video model has the extra img_emb.",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeCosmosPredict2_2B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("pos_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("x_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedding_norm.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(28):
            inputs.append(io.Float.Input(f"blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeCosmosPredict2_2B_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


class ModelMergeCosmosPredict2_14B(ModelMergeBlocks):
    @classmethod
    def define_schema(cls):
        inputs = [
            io.Model.Input("model1"),
            io.Model.Input("model2"),
            io.Float.Input("pos_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("x_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedder.", default=1.0, min=0.0, max=1.0, step=0.01),
            io.Float.Input("t_embedding_norm.", default=1.0, min=0.0, max=1.0, step=0.01)
        ]

        for i in range(36):
            inputs.append(io.Float.Input(f"blocks.{i}.", default=1.0, min=0.0, max=1.0, step=0.01))

        inputs.append(io.Float.Input("final_layer.", default=1.0, min=0.0, max=1.0, step=0.01))

        return io.Schema(
            node_id="ModelMergeCosmosPredict2_14B_V3",
            category="advanced/model_merging/model_specific",
            inputs=inputs,
            outputs=[
                io.Model.Output(),
            ]
        )


NODES_LIST: list[type[io.ComfyNode]] = [
    ModelMergeAuraflow,
    ModelMergeCosmos14B,
    ModelMergeCosmos7B,
    ModelMergeCosmosPredict2_14B,
    ModelMergeCosmosPredict2_2B,
    ModelMergeFlux1,
    ModelMergeLTXV,
    ModelMergeMochiPreview,
    ModelMergeSD1,
    ModelMergeSD3_2B,
    ModelMergeSD35_Large,
    ModelMergeSDXL,
    ModelMergeWAN2_1,
]
