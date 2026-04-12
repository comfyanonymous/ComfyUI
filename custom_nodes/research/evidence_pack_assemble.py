"""EvidencePackAssemble node - assemble evidence pack from claims and gaps."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class EvidencePackAssemble(io.ComfyNode):
    """Assemble evidence pack from claims and gap report."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EvidencePackAssemble",
            display_name="Assemble Evidence Pack",
            category="Research",
            inputs=[
                io.String.Input(
                    "gap_report",
                    display_name="Gap Report (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "claims_json",
                    display_name="Claims (JSON)",
                    default="[]",
                    multiline=True,
                ),
                io.String.Input(
                    "papers_json",
                    display_name="Papers (JSON)",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Evidence Pack (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, gap_report: str, claims_json: str, papers_json: str) -> io.NodeOutput:
        try:
            gap_data = json.loads(gap_report) if gap_report else {}
            claims = json.loads(claims_json) if claims_json else []
            papers = json.loads(papers_json) if papers_json else []
        except json.JSONDecodeError:
            gap_data = {}
            claims = []
            papers = []

        gaps = gap_data.get("gaps", [])
        evidence_pack = {"packs": [], "summary": {"total": 0, "by_gap": {}}}

        for gap in gaps:
            gap_id = gap.get("gap_id", f"gap_{len(evidence_pack['packs']) + 1}")
            gap_type = gap.get("gap_type", "")
            item_id = gap.get("item_id", "")

            # Find related claims
            related_claims = [
                c for c in claims
                if any(
                    c.get("claim_text", "").lower() in gap.get("description", "").lower()
                    for _ in [1]
                )
            ]

            # Find related papers
            related_papers = []
            for claim in related_claims:
                claim_text_lower = claim.get("claim_text", "").lower()
                for paper in papers:
                    abstract = paper.get("abstract", "").lower()
                    title = paper.get("title", "").lower()
                    if claim_text_lower in abstract or claim_text_lower in title:
                        related_papers.append(paper)

            pack = {
                "gap_id": gap_id,
                "item_id": item_id,
                "gap_type": gap_type,
                "severity": gap.get("severity", 1),
                "description": gap.get("description", ""),
                "related_claims": related_claims[:3],
                "related_papers": related_papers[:2],
                "evidence_strength": "strong" if len(related_claims) >= 2 else "weak",
            }

            evidence_pack["packs"].append(pack)
            key = f"{gap_type}"
            evidence_pack["summary"]["by_gap"][key] = evidence_pack["summary"]["by_gap"].get(key, 0) + 1

        evidence_pack["summary"]["total"] = len(evidence_pack["packs"])
        return io.NodeOutput(evidence_pack=json.dumps(evidence_pack, indent=2))
