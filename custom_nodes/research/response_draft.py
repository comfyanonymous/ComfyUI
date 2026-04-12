"""ResponseDraft node - draft rebuttal/response text."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ResponseDraft(io.ComfyNode):
    """Draft response text for review items."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ResponseDraft",
            display_name="Draft Response",
            category="Research",
            inputs=[
                io.String.Input(
                    "action_routes",
                    display_name="Action Routes (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "evidence_pack",
                    display_name="Evidence Pack (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "original_claims",
                    display_name="Original Claims (JSON)",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Drafted Responses (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, action_routes: str, evidence_pack: str, original_claims: str) -> io.NodeOutput:
        try:
            routes_data = json.loads(action_routes) if action_routes else {}
            evidence = json.loads(evidence_pack) if evidence_pack else {}
            claims = json.loads(original_claims) if original_claims else []
        except json.JSONDecodeError:
            routes_data = {}
            evidence = {}
            claims = []

        routes = routes_data.get("routes", [])
        packs = evidence.get("packs", [])

        responses = []

        for route in routes:
            gap_id = route.get("gap_id", "")
            action_type = route.get("action_type", "respond")
            description = route.get("description", "")

            # Find matching evidence pack
            matching_pack = next((p for p in packs if p.get("gap_id") == gap_id), None)

            if action_type == "experiment":
                response_text = (
                    f"We acknowledge the concern regarding {description}. "
                    f"We will conduct additional experiments as suggested "
                    f"and include the results in the revised manuscript."
                )
            elif action_type == "citation":
                response_text = (
                    f"We thank the reviewer for pointing out this gap. "
                    f"We will add relevant citations to address this concern."
                )
            elif action_type == "revise":
                claims_text = ""
                if matching_pack and matching_pack.get("related_claims"):
                    top_claim = matching_pack["related_claims"][0].get("claim_text", "")
                    claims_text = f" Our claim that '{top_claim[:50]}...' is supported by..."
                response_text = (
                    f"We appreciate the feedback. {claims_text} "
                    f"We will revise the manuscript to strengthen this argument."
                )
            else:
                response_text = (
                    f"We thank the reviewer for this comment. {description} "
                    f"We will address this concern in the revised manuscript."
                )

            responses.append({
                "gap_id": gap_id,
                "action_type": action_type,
                "response_text": response_text,
                "status": "drafted",
            })

        result = {
            "total_responses": len(responses),
            "responses": responses,
        }

        return io.NodeOutput(drafted_responses=json.dumps(result, indent=2))
