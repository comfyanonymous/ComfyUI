"""EvidenceGapDetect node - detect evidence gaps from review mappings."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class EvidenceGapDetect(io.ComfyNode):
    """Detect evidence gaps based on review item mappings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EvidenceGapDetect",
            display_name="Detect Evidence Gaps",
            category="Research",
            inputs=[
                io.String.Input(
                    "item_mappings",
                    display_name="Item Mappings (JSON)",
                    default="{}",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Gap Report (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, item_mappings: str) -> io.NodeOutput:
        try:
            data = json.loads(item_mappings) if item_mappings else {}
            mappings = data.get("mappings", [])
        except json.JSONDecodeError:
            mappings = []

        gaps = []
        gap_id = 1

        for mapping in mappings:
            severity = mapping.get("severity", 1)
            requires_experiment = mapping.get("requires_experiment", False)
            requires_citation = mapping.get("requires_citation", False)
            related_claims = mapping.get("related_claims", [])

            # Check for evidence gaps
            if requires_experiment and len(related_claims) == 0:
                gaps.append({
                    "gap_id": f"gap_{gap_id}",
                    "item_id": mapping.get("item_id"),
                    "gap_type": "missing_experiment",
                    "severity": severity,
                    "description": f"Review concern requires experimental evidence: {mapping.get('item_text', '')[:100]}...",
                    "suggested_action": "add_experiment",
                    "priority": "high" if severity >= 3 else "medium",
                })
                gap_id += 1

            if requires_citation and len(related_claims) == 0:
                gaps.append({
                    "gap_id": f"gap_{gap_id}",
                    "item_id": mapping.get("item_id"),
                    "gap_type": "missing_citation",
                    "severity": severity,
                    "description": f"Review concern requires citation to supporting work: {mapping.get('item_text', '')[:100]}...",
                    "suggested_action": "add_citation",
                    "priority": "medium",
                })
                gap_id += 1

            # Check for unsupported high-severity claims
            if severity >= 3 and len(related_claims) == 0:
                gaps.append({
                    "gap_id": f"gap_{gap_id}",
                    "item_id": mapping.get("item_id"),
                    "gap_type": "unsupported_major_claim",
                    "severity": severity,
                    "description": f"Major concern lacks supporting evidence: {mapping.get('item_text', '')[:100]}...",
                    "suggested_action": "strengthen_argument",
                    "priority": "high",
                })
                gap_id += 1

        # Summarize
        gap_summary = {
            "total_gaps": len(gaps),
            "high_priority": len([g for g in gaps if g["priority"] == "high"]),
            "medium_priority": len([g for g in gaps if g["priority"] == "medium"]),
            "by_type": {
                "missing_experiment": len([g for g in gaps if g["gap_type"] == "missing_experiment"]),
                "missing_citation": len([g for g in gaps if g["gap_type"] == "missing_citation"]),
                "unsupported_major_claim": len([g for g in gaps if g["gap_type"] == "unsupported_major_claim"]),
            },
            "gaps": gaps,
        }

        return io.NodeOutput(gap_report=json.dumps(gap_summary, indent=2))