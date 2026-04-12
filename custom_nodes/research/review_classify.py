# custom_nodes/research/review_classify.py
"""ReviewClassify node - classify review items by type and severity."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ReviewClassify(io.ComfyNode):
    """Classify review items by type and severity."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReviewClassify",
            display_name="Classify Review Items",
            category="Research",
            inputs=[
                io.String.Input(
                    "review_items",
                    display_name="Review Items (JSON)",
                    default="{}",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Classified Items (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, review_items: str) -> io.NodeOutput:
        try:
            data = json.loads(review_items) if review_items else {}
            items = data.get("items", [])
        except json.JSONDecodeError:
            items = []

        # Define categories
        categories = {
            "methodology": {"icon": "🔬", "color": "#388bfd"},
            "clarity": {"icon": "📝", "color": "#d29922"},
            "error": {"icon": "❌", "color": "#f85149"},
            "missing_info": {"icon": "📋", "color": "#a371f7"},
            "major_concern": {"icon": "⚠️", "color": "#f85149"},
            "suggestion": {"icon": "💡", "color": "#2ea043"},
            "reference": {"icon": "📚", "color": "#58a6ff"},
            "general": {"icon": "💬", "color": "#8b949e"},
        }

        # Classify each item
        classified = []
        for item in items:
            item_type = item.get("type", "general")
            severity = item.get("severity", 1)

            # Determine action category
            if severity >= 3:
                action = "needs_revision"
            elif severity == 2:
                action = "needs_response"
            else:
                action = "consider"

            # Map to manuscript section
            section_map = {
                "methodology": "Methods",
                "error": "Methods",
                "clarity": "Writing",
                "missing_info": "Introduction",
                "reference": "Related Work",
            }

            classified_item = {
                **item,
                "category": item_type,
                "action": action,
                "target_section": section_map.get(item_type, "General"),
                "icon": categories.get(item_type, categories["general"])["icon"],
                "color": categories.get(item_type, categories["general"])["color"],
                "priority": "high" if severity >= 3 else "medium" if severity == 2 else "low",
            }
            classified.append(classified_item)

        result = {
            "total_items": len(classified),
            "by_severity": {
                "high": len([i for i in classified if i["priority"] == "high"]),
                "medium": len([i for i in classified if i["priority"] == "medium"]),
                "low": len([i for i in classified if i["priority"] == "low"]),
            },
            "by_action": {
                "needs_revision": len([i for i in classified if i["action"] == "needs_revision"]),
                "needs_response": len([i for i in classified if i["action"] == "needs_response"]),
                "consider": len([i for i in classified if i["action"] == "consider"]),
            },
            "items": classified,
        }

        return io.NodeOutput(classified_items=json.dumps(result, indent=2))