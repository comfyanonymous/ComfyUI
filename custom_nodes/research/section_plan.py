"""SectionPlan node - generate section narrative plan."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class SectionPlan(io.ComfyNode):
    """Generate a narrative plan for a manuscript section based on claims and style."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SectionPlan",
            display_name="Section Plan",
            category="Research",
            inputs=[
                io.String.Input(
                    "section_type",
                    display_name="Section Type",
                    default="abstract",
                ),
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
            ],
            outputs=[
                io.String.Output(display_name="Section Plan (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, section_type: str, claims: str, style_profile: str) -> io.NodeOutput:
        try:
            claims_list = json.loads(claims) if claims else []
            style = json.loads(style_profile) if style_profile else {}
        except json.JSONDecodeError:
            claims_list = []
            style = {}

        # Generate plan based on section type
        plan = {
            "section_type": section_type,
            "paragraphs": [],
            "estimated_length": "",
        }

        if section_type.lower() == "abstract":
            plan["paragraphs"] = [
                {"order": 1, "purpose": "Background and problem statement", "key_claims": []},
                {"order": 2, "purpose": "Proposed method", "key_claims": []},
                {"order": 3, "purpose": "Key results", "key_claims": []},
                {"order": 4, "purpose": "Conclusion and significance", "key_claims": []},
            ]
            plan["estimated_length"] = "150-250 words"
        elif section_type.lower() == "introduction":
            plan["paragraphs"] = [
                {"order": 1, "purpose": "Hook - why this matters", "key_claims": []},
                {"order": 2, "purpose": "Background and related work", "key_claims": []},
                {"order": 3, "purpose": "Gap in existing approaches", "key_claims": []},
                {"order": 4, "purpose": "Our contribution", "key_claims": []},
            ]
            plan["estimated_length"] = "400-800 words"
        elif section_type.lower() == "methods":
            plan["paragraphs"] = [
                {"order": 1, "purpose": "Overview of approach", "key_claims": []},
                {"order": 2, "purpose": "Dataset and preprocessing", "key_claims": []},
                {"order": 3, "purpose": "Model architecture", "key_claims": []},
                {"order": 4, "purpose": "Training procedure", "key_claims": []},
                {"order": 5, "purpose": "Evaluation metrics", "key_claims": []},
            ]
            plan["estimated_length"] = "600-1000 words"

        # Associate claims with paragraphs
        for i, claim in enumerate(claims_list[:4]):
            if i < len(plan["paragraphs"]):
                plan["paragraphs"][i]["key_claims"].append(claim.get("text", ""))

        plan["style_notes"] = style.get("abstract_pattern", "standard")

        return io.NodeOutput(section_plan=json.dumps(plan, indent=2))
