"""ExportManuscript node - compile manuscript sections to output format."""
import json
import os
from datetime import datetime
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ExportManuscript(io.ComfyNode):
    """Compile manuscript sections into a formatted document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ExportManuscript",
            display_name="Export Manuscript",
            category="Research",
            inputs=[
                io.String.Input(
                    "title",
                    display_name="Paper Title",
                    default="",
                ),
                io.String.Input(
                    "authors",
                    display_name="Authors (comma-separated)",
                    default="",
                ),
                io.String.Input(
                    "abstract",
                    display_name="Abstract",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "introduction",
                    display_name="Introduction",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "methods",
                    display_name="Methods",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "results",
                    display_name="Results",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "conclusion",
                    display_name="Conclusion",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "output_format",
                    display_name="Output Format",
                    default="markdown",
                ),
            ],
            outputs=[
                io.String.Output(display_name="Output Path"),
                io.String.Output(display_name="Manuscript Text"),
            ],
        )

    @classmethod
    def execute(cls, title: str, authors: str, abstract: str, introduction: str,
                methods: str, results: str, conclusion: str, output_format: str) -> io.NodeOutput:

        # Build manuscript
        lines = []

        if title:
            lines.append(f"# {title}")
            lines.append("")

        if authors:
            author_list = [a.strip() for a in authors.split(",")]
            lines.append("**Authors:** " + ", ".join(author_list))
            lines.append("")

        lines.append("---")
        lines.append("")

        if abstract:
            lines.append("## Abstract")
            lines.append("")
            lines.append(abstract)
            lines.append("")

        if introduction:
            lines.append("## 1. Introduction")
            lines.append("")
            lines.append(introduction)
            lines.append("")

        if methods:
            lines.append("## 2. Methods")
            lines.append("")
            lines.append(methods)
            lines.append("")

        if results:
            lines.append("## 3. Results")
            lines.append("")
            lines.append(results)
            lines.append("")

        if conclusion:
            lines.append("## 4. Conclusion")
            lines.append("")
            lines.append(conclusion)
            lines.append("")

        manuscript_text = "\n".join(lines)

        # Save to file
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "research_workbench", "exports")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manuscript_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(manuscript_text)
            output_path = filepath
        except Exception as e:
            output_path = f"Error saving: {str(e)}"

        return io.NodeOutput(
            output_path=output_path,
            manuscript_text=manuscript_text[:5000]  # Truncate for display
        )
