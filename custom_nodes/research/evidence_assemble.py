"""ClaimEvidenceAssemble node - assemble claim × evidence matrix."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ClaimEvidenceAssemble(io.ComfyNode):
    """Assemble claims and papers into a structured evidence matrix."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ClaimEvidenceAssemble",
            display_name="Assemble Evidence",
            category="Research",
            inputs=[
                io.String.Input(
                    "claims",
                    display_name="Claims (JSON)",
                    default="[]",
                    multiline=True,
                ),
                io.String.Input(
                    "papers",
                    display_name="Papers (JSON)",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Evidence Matrix (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, claims: str, papers: str) -> io.NodeOutput:
        try:
            claims_list = json.loads(claims) if claims else []
            papers_list = json.loads(papers) if papers else []

            matrix = []
            for claim in claims_list:
                claim_text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
                matrix.append({
                    "claim": claim_text,
                    "claim_type": claim.get("type", "") if isinstance(claim, dict) else "",
                    "support_level": claim.get("support_level", "unsupported") if isinstance(claim, dict) else "unsupported",
                    "evidence": [],
                    "gap_flags": ["No linked evidence yet"],
                })

            return io.NodeOutput(evidence_matrix=json.dumps(matrix, indent=2))
        except json.JSONDecodeError as e:
            return io.NodeOutput(evidence_matrix=json.dumps({"error": f"JSON parse error: {e}"}))
