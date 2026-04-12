"""ReferencePaperSelect node - select papers from project for canvas use."""
import json
from typing_extensions import override

from comfy_api.latest import ComfyNode, io


class ReferencePaperSelect(io.ComfyNode):
    """Select papers from a project to load onto the canvas for reference."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReferencePaperSelect",
            display_name="Select Papers",
            category="Research",
            inputs=[
                io.String.Input(
                    "project_id",
                    display_name="Project ID",
                    default="",
                ),
                io.Int.Input(
                    "max_papers",
                    display_name="Max Papers",
                    default=5,
                    min=1,
                    max=20,
                    step=1,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Selected Papers (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, project_id: str, max_papers: int) -> io.NodeOutput:
        if not project_id.strip():
            return io.NodeOutput(selected_papers=json.dumps([]))

        # Phase 1: Returns mock data
        # Phase 2+ would query the actual project papers from DB
        return io.NodeOutput(selected_papers=json.dumps([
            {
                "paper_id": "mock-id-1",
                "title": "Sample Paper 1",
                "authors": ["Author A", "Author B"],
                "abstract": "This is a sample abstract for demonstration.",
                "year": "2024",
            },
            {
                "paper_id": "mock-id-2",
                "title": "Sample Paper 2",
                "authors": ["Author C"],
                "abstract": "Another sample abstract.",
                "year": "2023",
            }
        ][:max_papers], indent=2))
