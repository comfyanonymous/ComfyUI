# custom_nodes/research/review_map.py
"""ReviewMap node - map review items to manuscript sections/claims."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ReviewMap(io.ComfyNode):
    """Map review items to claims, sections, and figures in the manuscript."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReviewMap",
            display_name="Map to Manuscript",
            category="Research",
            inputs=[
                io.String.Input(
                    "classified_items",
                    display_name="Classified Items (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "manuscript_sections",
                    display_name="Manuscript Sections (JSON)",
                    default="[]",
                    multiline=True,
                ),
                io.String.Input(
                    "claims",
                    display_name="Claims (JSON)",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Item Mappings (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, classified_items: str, manuscript_sections: str, claims: str) -> io.NodeOutput:
        try:
            data = json.loads(classified_items) if classified_items else {}
            items = data.get("items", [])
        except json.JSONDecodeError:
            items = []

        try:
            sections = json.loads(manuscript_sections) if manuscript_sections else []
        except json.JSONDecodeError:
            sections = []

        try:
            claims_list = json.loads(claims) if claims else []
        except json.JSONDecodeError:
            claims_list = []

        mappings = []

        for item in items:
            item_text = item.get("text", "").lower()
            item_type = item.get("type", "general")

            # Find related claims
            related_claims = []
            for claim in claims_list:
                claim_text = claim.get("text", "").lower()
                # Simple keyword overlap
                item_words = set(item_text.split())
                claim_words = set(claim_text.split())
                overlap = item_words & claim_words
                if len(overlap) >= 3:
                    related_claims.append(claim.get("text", "")[:100])

            # Find target section
            target_section = item.get("target_section", "General")

            mapping = {
                "item_id": item.get("id"),
                "item_text": item.get("text", ""),
                "severity": item.get("severity"),
                "target_section": target_section,
                "related_claims": related_claims[:3],
                "requires_citation": "citation" in item_text or "cite" in item_text,
                "requires_experiment": any(w in item_text for w in ["experiment", "ablation", "baseline", "comparison"]),
                "response_strategy": item.get("action", "consider"),
            }
            mappings.append(mapping)

        result = {
            "total_mappings": len(mappings),
            "requires_revision": len([m for m in mappings if m["requires_experiment"]]),
            "requires_citation": len([m for m in mappings if m["requires_citation"]]),
            "mappings": mappings,
        }

        return io.NodeOutput(item_mappings=json.dumps(result, indent=2))