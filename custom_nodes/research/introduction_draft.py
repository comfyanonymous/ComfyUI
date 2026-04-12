"""IntroductionDraft node - generate introduction text."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class IntroductionDraft(io.ComfyNode):
    """Generate an introduction draft based on problem framing and contributions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IntroductionDraft",
            display_name="Draft Introduction",
            category="Research",
            inputs=[
                io.String.Input(
                    "problem_framing",
                    display_name="Problem Framing",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "gap_statement",
                    display_name="Gap Statement",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "contribution",
                    display_name="Our Contribution",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "style_profile",
                    display_name="Style Profile (JSON)",
                    default="{}",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Introduction Text"),
            ],
        )

    @classmethod
    def execute(cls, problem_framing: str, gap_statement: str, contribution: str, style_profile: str) -> io.NodeOutput:
        try:
            style = json.loads(style_profile) if style_profile else {}
        except json.JSONDecodeError:
            style = {}

        paragraphs = []

        # Paragraph 1: Hook and problem
        if problem_framing:
            paragraphs.append(f"{problem_framing}")
        else:
            paragraphs.append("Medical image analysis plays a crucial role in modern healthcare. Accurate segmentation of anatomical structures enables precise diagnosis and treatment planning. However, achieving reliable automated segmentation remains challenging due to anatomical variability and imaging artifacts.")

        # Paragraph 2: Gap
        if gap_statement:
            paragraphs.append(f"{gap_statement}")
        else:
            paragraphs.append("Existing approaches often struggle with boundary ambiguity and fail to generalize across different imaging protocols. While deep learning methods have shown promise, they typically require large annotated datasets that are expensive to obtain in medical domains.")

        # Paragraph 3: Contribution
        if contribution:
            paragraphs.append(f"{contribution}")
        else:
            paragraphs.append("To address these limitations, we propose a novel framework that leverages self-supervised pretraining and domain adaptation. Our approach reduces annotation requirements while improving segmentation accuracy across diverse imaging conditions.")

        # Paragraph 4: Structure
        paragraphs.append("The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes our methodology. Section 4 presents experimental results. Section 5 discusses implications and concludes the paper.")

        intro_text = "\n\n".join(paragraphs)

        return io.NodeOutput(intro_text=intro_text)
