"""AbstractDraft node - generate abstract text."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class AbstractDraft(io.ComfyNode):
    """Generate an abstract draft based on claims and style profile."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AbstractDraft",
            display_name="Draft Abstract",
            category="Research",
            inputs=[
                io.String.Input(
                    "claims",
                    display_name="Claims (JSON)",
                    default="[]",
                    multiline=True,
                ),
                io.String.Input(
                    "style_profile",
                    display_name="Style Profile (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.Int.Input(
                    "max_words",
                    display_name="Max Words",
                    default=250,
                    min=100,
                    max=400,
                    step=10,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Abstract Text"),
            ],
        )

    @classmethod
    def execute(cls, claims: str, style_profile: str, max_words: int) -> io.NodeOutput:
        try:
            claims_list = json.loads(claims) if claims else []
            style = json.loads(style_profile) if style_profile else {}
        except json.JSONDecodeError:
            claims_list = []
            style = {}

        # Extract key information from claims
        performance_claims = [c for c in claims_list if c.get("type") == "performance"]
        method_claims = [c for c in claims_list if c.get("type") == "method"]

        sentences = []

        # Background sentence
        sentences.append("Medical image segmentation remains a critical challenge in clinical practice.")

        # Method sentence
        if method_claims:
            method_text = method_claims[0].get("text", "We propose a novel approach.")
            sentences.append(f"In this work, {method_text}")
        else:
            sentences.append("We present a deep learning framework for automated segmentation.")

        # Results sentence
        if performance_claims:
            perf_text = performance_claims[0].get("text", "Our method achieves strong performance.")
            sentences.append(f"Experimental results demonstrate that {perf_text}")
        else:
            sentences.append("Our approach achieves competitive performance on benchmark datasets.")

        # Conclusion sentence
        sentences.append("These findings suggest potential for clinical application.")

        abstract = " ".join(sentences)

        # Apply style if available
        tone = style.get("tone_notes", "neutral")
        if tone == "enthusiastic":
            abstract = abstract.replace("competitive", "state-of-the-art").replace("strong", "exceptional")

        return io.NodeOutput(abstract_text=abstract)
