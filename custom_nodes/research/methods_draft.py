"""MethodsDraft node - generate methods section text."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class MethodsDraft(io.ComfyNode):
    """Generate a methods section draft based on claims and data assets."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MethodsDraft",
            display_name="Draft Methods",
            category="Research",
            inputs=[
                io.String.Input(
                    "approach_overview",
                    display_name="Approach Overview",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "dataset_description",
                    display_name="Dataset Description",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "model_architecture",
                    display_name="Model Architecture",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "training_details",
                    display_name="Training Details",
                    default="",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Methods Text"),
            ],
        )

    @classmethod
    def execute(cls, approach_overview: str, dataset_description: str, model_architecture: str, training_details: str) -> io.NodeOutput:

        sections = []

        # 3.1 Overview
        sections.append("3.1 Overview")
        if approach_overview:
            sections.append(approach_overview)
        else:
            sections.append("Our method consists of three main components: (1) a preprocessing module for standardizing input images, (2) a deep neural network for feature extraction and segmentation, and (3) a post-processing step for refining predictions.")

        # 3.2 Dataset
        sections.append("\n3.2 Dataset and Preprocessing")
        if dataset_description:
            sections.append(dataset_description)
        else:
            sections.append("We evaluate our method on a dataset comprising medical images from multiple institutions. Images are resampled to a common resolution and intensity-normalized. Data augmentation including random rotation, scaling, and elastic deformation is applied during training.")

        # 3.3 Model Architecture
        sections.append("\n3.3 Model Architecture")
        if model_architecture:
            sections.append(model_architecture)
        else:
            sections.append("The network architecture follows an encoder-decoder design with skip connections. The encoder extracts multi-scale features using residual blocks, while the decoder progressively upsamples feature maps to produce segmentation masks. Attention mechanisms are incorporated to focus on relevant regions.")

        # 3.4 Training
        sections.append("\n3.4 Training Procedure")
        if training_details:
            sections.append(training_details)
        else:
            sections.append("Models are trained using the Adam optimizer with a learning rate of 1e-4. We employ a combined loss function consisting of Dice loss and cross-entropy. Training is performed for 200 epochs with early stopping based on validation performance.")

        # 3.5 Evaluation
        sections.append("\n3.5 Evaluation Metrics")
        sections.append("Segmentation performance is evaluated using Dice similarity coefficient (DSC), Intersection over Union (IoU), and Hausdorff distance. Statistical significance is assessed using paired t-tests with Bonferroni correction for multiple comparisons.")

        methods_text = "\n".join(sections)

        return io.NodeOutput(methods_text=methods_text)
