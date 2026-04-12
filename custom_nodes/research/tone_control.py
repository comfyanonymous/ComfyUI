"""ToneControl node - adjust tone of response text."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ToneControl(io.ComfyNode):
    """Adjust tone of drafted responses."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ToneControl",
            display_name="Tone Control",
            category="Research",
            inputs=[
                io.String.Input(
                    "drafted_responses",
                    display_name="Drafted Responses (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "tone",
                    display_name="Target Tone",
                    default="professional",
                ),
            ],
            outputs=[
                io.String.Output(display_name="Finalized Responses (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, drafted_responses: str, tone: str) -> io.NodeOutput:
        try:
            data = json.loads(drafted_responses) if drafted_responses else {}
        except json.JSONDecodeError:
            data = {}

        responses = data.get("responses", [])

        tone_prefixes = {
            "apologetic": "We sincerely apologize for...",
            "professional": "We thank the reviewer for",
            "confident": "We stand by our",
            "diplomatic": "We appreciate the feedback and will",
        }

        tone_suffixes = {
            "apologetic": "We are committed to improving this.",
            "professional": "We will address this in the revision.",
            "confident": "This is well-supported by our results.",
            "diplomatic": "We will carefully consider this suggestion.",
        }

        prefix = tone_prefixes.get(tone.lower(), "We thank the reviewer for")
        suffix = tone_suffixes.get(tone.lower(), "We will address this in the revision.")

        finalized = []

        for resp in responses:
            text = resp.get("response_text", "")
            action_type = resp.get("action_type", "")

            # Apply tone transformations
            if tone.lower() == "apologetic":
                text = f"{prefix} {text.lower()}"
                text = text.rstrip(".") + f". {suffix}"
            elif tone.lower() == "confident":
                if "acknowledge" in text.lower():
                    text = text.replace("acknowledge", "note")
                if "will revise" in text.lower():
                    text = text.replace("will revise", "have revised")
            elif tone.lower() == "diplomatic":
                text = text.replace("We will", "We would be happy to")
            else:  # professional
                if not text.startswith("We "):
                    text = f"{prefix} {text}"

            finalized.append({
                **resp,
                "response_text": text,
                "tone_applied": tone,
                "status": "finalized",
            })

        result = {
            "total": len(finalized),
            "tone": tone,
            "responses": finalized,
        }

        return io.NodeOutput(finalized_responses=json.dumps(result, indent=2))
