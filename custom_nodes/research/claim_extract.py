"""PaperClaimExtract node - extract claims from paper text."""
import json
import re
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class PaperClaimExtract(io.ComfyNode):
    """Extract scientific claims from paper text or abstract."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PaperClaimExtract",
            display_name="Extract Claims",
            category="Research",
            inputs=[
                io.String.Input(
                    "paper_text",
                    display_name="Paper Text / Abstract",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "claim_types",
                    display_name="Claim Types (comma-separated)",
                    default="performance,robustness,generalization",
                ),
            ],
            outputs=[
                io.String.Output(display_name="Claims (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, paper_text: str, claim_types: str) -> io.NodeOutput:
        if not paper_text.strip():
            return io.NodeOutput(claims=json.dumps([]))

        claim_type_list = [t.strip() for t in claim_types.split(",")]
        sentences = re.split(r'(?<=[.!?])\s+', paper_text.strip())

        claims = []
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 20:
                continue
            lower = sent.lower()
            # Heuristic: sentences with comparison/performance words
            if any(w in lower for w in ["achieves", "outperforms", "improves", "reduces", "increases", "demonstrates", "shows", "provides", "enables"]):
                claims.append({
                    "text": sent,
                    "type": _classify_claim(sent, claim_type_list),
                    "support_level": "unsupported",
                })

        return io.NodeOutput(claims=json.dumps(claims[:10], indent=2))  # limit to 10


def _classify_claim(sentence: str, claim_types) -> str:
    lower = sentence.lower()
    if any(w in lower for w in ["accuracy", "performance", "score", "auc", "f1", "precision", "recall"]):
        return "performance"
    if any(w in lower for w in ["robust", "noise", "attack", "adversarial", "corruption"]):
        return "robustness"
    if any(w in lower for w in ["general", "transfer", "cross-domain", "cross-dataset"]):
        return "generalization"
    return claim_types[0] if claim_types else "other"
