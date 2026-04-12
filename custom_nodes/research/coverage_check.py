"""CoverageCheck node - check response coverage of review items."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class CoverageCheck(io.ComfyNode):
    """Check if responses adequately cover all review items."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CoverageCheck",
            display_name="Coverage Check",
            category="Research",
            inputs=[
                io.String.Input(
                    "finalized_responses",
                    display_name="Finalized Responses (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "review_items",
                    display_name="Review Items (JSON)",
                    default="[]",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Coverage Report (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, finalized_responses: str, review_items: str) -> io.NodeOutput:
        try:
            responses_data = json.loads(finalized_responses) if finalized_responses else {}
            items = json.loads(review_items) if review_items else []
        except json.JSONDecodeError:
            responses_data = {}
            items = []

        responses = responses_data.get("responses", [])

        coverage = []
        covered_ids = set()
        uncovered_items = []

        for resp in responses:
            gap_id = resp.get("gap_id", "")
            response_text = resp.get("response_text", "")
            action_type = resp.get("action_type", "")

            coverage.append({
                "gap_id": gap_id,
                "covered": True,
                "response_length": len(response_text),
                "action_type": action_type,
                "assessment": "adequate" if len(response_text) > 50 else "too_brief",
            })
            covered_ids.add(gap_id)

        # Check for uncovered items
        for item in items:
            item_id = item.get("item_id", f"item_{len(coverage)}")
            if item_id not in covered_ids:
                uncovered_items.append({
                    "item_id": item_id,
                    "issue": item.get("item_text", "")[:100],
                    "severity": item.get("severity", 1),
                    "status": "not_addressed",
                })

        # Determine overall coverage
        total = len(coverage) + len(uncovered_items)
        covered_count = len(coverage)
        coverage_pct = (covered_count / total * 100) if total > 0 else 100

        report = {
            "total_items": total,
            "covered_items": covered_count,
            "uncovered_items": len(uncovered_items),
            "coverage_percentage": round(coverage_pct, 1),
            "overall_status": "complete" if coverage_pct >= 90 else "partial" if coverage_pct >= 50 else "incomplete",
            "item_coverage": coverage,
            "uncovered": uncovered_items,
        }

        return io.NodeOutput(coverage_report=json.dumps(report, indent=2))
