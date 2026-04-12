"""RevisionPlan node - create revision task plan from coverage report."""
import json
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class RevisionPlan(io.ComfyNode):
    """Create a structured revision task plan from coverage report."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RevisionPlan",
            display_name="Revision Plan",
            category="Research",
            inputs=[
                io.String.Input(
                    "coverage_report",
                    display_name="Coverage Report (JSON)",
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "action_routes",
                    display_name="Action Routes (JSON)",
                    default="{}",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Revision Plan (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, coverage_report: str, action_routes: str) -> io.NodeOutput:
        try:
            coverage = json.loads(coverage_report) if coverage_report else {}
            routes_data = json.loads(action_routes) if action_routes else {}
        except json.JSONDecodeError:
            coverage = {}
            routes_data = {}

        uncovered = coverage.get("uncovered_items", [])
        routes = routes_data.get("routes", [])
        item_coverage = coverage.get("item_coverage", [])

        # Group by action type
        by_action = {"experiment": [], "citation": [], "revise": [], "respond": []}
        for route in routes:
            action_type = route.get("action_type", "respond")
            if action_type in by_action:
                by_action[action_type].append(route)

        # Create revision tasks
        tasks = []
        task_id = 1

        # Priority 1: High severity uncovered
        for item in uncovered:
            if item.get("severity", 1) >= 3:
                tasks.append({
                    "task_id": f"task_{task_id}",
                    "priority": "high",
                    "action": "address_immediately",
                    "description": f"Uncovered high-severity item: {item.get('issue', '')[:80]}",
                    "estimated_time": "2-4 hours",
                })
                task_id += 1

        # Priority 2: Experiment tasks
        for route in by_action.get("experiment", []):
            tasks.append({
                "task_id": f"task_{task_id}",
                "priority": "high",
                "action": "run_experiment",
                "description": route.get("description", ""),
                "effort": route.get("estimated_effort", "high"),
                "action_items": route.get("action_items", []),
            })
            task_id += 1

        # Priority 3: Citation tasks
        for route in by_action.get("citation", []):
            tasks.append({
                "task_id": f"task_{task_id}",
                "priority": "medium",
                "action": "add_citations",
                "description": route.get("description", ""),
                "effort": route.get("estimated_effort", "low"),
                "action_items": route.get("action_items", []),
            })
            task_id += 1

        # Priority 4: Revise tasks
        for route in by_action.get("revise", []):
            tasks.append({
                "task_id": f"task_{task_id}",
                "priority": "medium",
                "action": "revise_manuscript",
                "description": route.get("description", ""),
                "effort": route.get("estimated_effort", "medium"),
                "action_items": route.get("action_items", []),
            })
            task_id += 1

        # Priority 5: Response tasks
        for route in by_action.get("respond", []):
            tasks.append({
                "task_id": f"task_{task_id}",
                "priority": "low",
                "action": "write_response",
                "description": route.get("description", ""),
                "effort": route.get("estimated_effort", "medium"),
                "action_items": route.get("action_items", []),
            })
            task_id += 1

        # Add brief uncovered items
        for item in uncovered:
            if item.get("severity", 1) < 3:
                tasks.append({
                    "task_id": f"task_{task_id}",
                    "priority": "low",
                    "action": "review_uncovered",
                    "description": f"Review item: {item.get('issue', '')[:80]}",
                    "estimated_time": "30 minutes",
                })
                task_id += 1

        # Summarize
        summary = {
            "total_tasks": len(tasks),
            "by_priority": {
                "high": len([t for t in tasks if t.get("priority") == "high"]),
                "medium": len([t for t in tasks if t.get("priority") == "medium"]),
                "low": len([t for t in tasks if t.get("priority") == "low"]),
            },
            "by_action": {
                "run_experiment": len(by_action.get("experiment", [])),
                "add_citations": len(by_action.get("citation", [])),
                "revise_manuscript": len(by_action.get("revise", [])),
                "write_response": len(by_action.get("respond", [])),
            },
        }

        plan = {
            "summary": summary,
            "tasks": tasks,
            "estimated_total_time": _estimate_time(tasks),
        }

        return io.NodeOutput(revision_plan=json.dumps(plan, indent=2))


def _estimate_time(tasks: list) -> str:
    """Estimate total time for all tasks."""
    hours = 0
    for task in tasks:
        time_str = task.get("estimated_time", "1 hour")
        if "hour" in time_str:
            try:
                hours += float(time_str.split()[0])
            except (ValueError, IndexError):
                hours += 1
        elif "minute" in time_str:
            try:
                hours += float(time_str.split()[0]) / 60
            except (ValueError, IndexError):
                pass
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    return f"{int(hours)}-{int(hours + 2)} hours"
