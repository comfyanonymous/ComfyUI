# custom_nodes/research/review_atomize.py
"""ReviewAtomize node - split review into atomic items."""
import json
import re
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ReviewAtomize(io.ComfyNode):
    """Split review text into atomic feedback items."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReviewAtomize",
            display_name="Atomize Review",
            category="Research",
            inputs=[
                io.String.Input(
                    "raw_review",
                    display_name="Raw Review (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.Int.Input(
                    "min_severity",
                    display_name="Min Severity to Extract",
                    default=1,
                    min=1,
                    max=3,
                    step=1,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Review Items (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, raw_review: str, min_severity: int) -> io.NodeOutput:
        try:
            review = json.loads(raw_review) if raw_review else {}
        except json.JSONDecodeError:
            review = {"original_text": raw_review}

        original_text = review.get("original_text", "")
        paragraphs = review.get("paragraphs", [])

        items = []
        item_id = 1

        # Common reviewer phrases that indicate feedback
        feedback_patterns = [
            (r"the (?:paper|method|approach|model|results?|experiments?) (?:is|are|should|could|may|might) (.+)", "methodology"),
            (r"(?:it is|this is|there is) (?:not|rarely|insufficiently) (.+)", "gap"),
            (r"(?:more|additional|further) (.+) (?:is|are) needed", "missing"),
            (r"(?:unclear|confusing|vague|ambiguous) (.+)", "clarity"),
            (r"(?:incorrect|inaccurate|wrong|error) (.+)", "error"),
            (r"(?:major|significant|critical|important) (.+)", "major"),
            (r"(?:minor|small|small) (.+)", "minor"),
            (r"see (?:Section|Figure|Table) (\d+)", "reference"),
        ]

        # Split by sentence and extract items
        sentences = re.split(r'(?<=[.!?])\s+', original_text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue

            # Classify the item
            item_type = "general"
            severity = 1

            sentence_lower = sentence.lower()

            if any(w in sentence_lower for w in ["major", "significant", "critical", "important", "must"]):
                severity = 3
                item_type = "major_concern"
            elif any(w in sentence_lower for w in ["minor", "small", "suggestion"]):
                severity = 1
                item_type = "suggestion"
            elif any(w in sentence_lower for w in ["unclear", "confusing", "vague", "ambiguous"]):
                severity = 2
                item_type = "clarity"
            elif any(w in sentence_lower for w in ["error", "incorrect", "wrong", "mistake"]):
                severity = 3
                item_type = "error"
            elif any(w in sentence_lower for w in ["missing", "lacking", "insufficient", "not enough"]):
                severity = 2
                item_type = "missing_info"

            if severity >= min_severity:
                items.append({
                    "id": f"item_{item_id}",
                    "text": sentence,
                    "type": item_type,
                    "severity": severity,
                    "quoted_claims": [],
                })
                item_id += 1

        result = {
            "total_items": len(items),
            "items": items,
        }

        return io.NodeOutput(review_items=json.dumps(result, indent=2))
