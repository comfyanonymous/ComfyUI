"""ActionRoute node - route gaps to appropriate response actions."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ActionRoute(io.ComfyNode):
    """Route review gaps to Respond/Revise/Citation/Experiment actions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ActionRoute",
            display_name="Route Actions",
            category="Research",
            inputs=[
                io.String.Input(
                    "gap_report",
                    display_name="Gap Report (JSON)",
                    default="{}",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Action Routes (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, gap_report: str) -> io.NodeOutput:
        try:
            data = json.loads(gap_report) if gap_report else {}
            gaps = data.get("gaps", [])
        except json.JSONDecodeError:
            gaps = []

        routes = []

        for gap in gaps:
            gap_type = gap.get("gap_type", "")
            suggested_action = gap.get("suggested_action", "")

            # Determine route
            if gap_type == "missing_experiment" or suggested_action == "add_experiment":
                route = {
                    "gap_id": gap.get("gap_id"),
                    "action_type": "experiment",
                    "route": "perform_experiment",
                    "description": f"Perform additional experiments to address: {gap.get('description', '')[:80]}...",
                    "estimated_effort": "high",
                    "action_items": [
                        "Design experiment protocol",
                        "Run experiments",
                        "Analyze results",
                        "Update manuscript",
                    ],
                }
            elif gap_type == "missing_citation" or suggested_action == "add_citation":
                route = {
                    "gap_id": gap.get("gap_id"),
                    "action_type": "citation",
                    "route": "add_citation",
                    "description": f"Add citation to address: {gap.get('description', '')[:80]}...",
                    "estimated_effort": "low",
                    "action_items": [
                        "Search for relevant papers",
                        "Add citations",
                        "Update reference list",
                    ],
                }
            elif suggested_action == "strengthen_argument":
                route = {
                    "gap_id": gap.get("gap_id"),
                    "action_type": "revise",
                    "route": "revise_manuscript",
                    "description": f"Revise manuscript to address: {gap.get('description', '')[:80]}...",
                    "estimated_effort": "medium",
                    "action_items": [
                        "Rewrite affected sections",
                        "Add supporting discussion",
                        "Verify consistency",
                    ],
                }
            else:
                route = {
                    "gap_id": gap.get("gap_id"),
                    "action_type": "respond",
                    "route": "write_response",
                    "description": f"Write response to: {gap.get('description', '')[:80]}...",
                    "estimated_effort": "medium",
                    "action_items": [
                        "Draft response text",
                        "Gather supporting evidence",
                        "Finalize response",
                    ],
                }

            routes.append(route)

        # Group by action type
        by_action = {
            "experiment": [r for r in routes if r["action_type"] == "experiment"],
            "citation": [r for r in routes if r["action_type"] == "citation"],
            "revise": [r for r in routes if r["action_type"] == "revise"],
            "respond": [r for r in routes if r["action_type"] == "respond"],
        }

        result = {
            "total_routes": len(routes),
            "by_action": {k: len(v) for k, v in by_action.items()},
            "routes": routes,
        }

        return io.NodeOutput(action_routes=json.dumps(result, indent=2))