# custom_nodes/research/review_import.py
"""ReviewImport node - import review text."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ReviewImport(io.ComfyNode):
    """Import reviewer feedback text for analysis."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReviewImport",
            display_name="Import Review",
            category="Research",
            inputs=[
                io.String.Input(
                    "review_text",
                    display_name="Review Text",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "reviewer_role",
                    display_name="Reviewer Role",
                    default="reviewer",
                ),
            ],
            outputs=[
                io.String.Output(display_name="Raw Review (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, review_text: str, reviewer_role: str) -> io.NodeOutput:
        if not review_text.strip():
            return io.NodeOutput(raw_review=json.dumps({"error": "No review text provided"}))

        # Parse the review into structured format
        raw_review = {
            "reviewer_role": reviewer_role,
            "original_text": review_text,
            "paragraphs": [p.strip() for p in review_text.split("\n\n") if p.strip()],
            "word_count": len(review_text.split()),
        }

        return io.NodeOutput(raw_review=json.dumps(raw_review, indent=2))
