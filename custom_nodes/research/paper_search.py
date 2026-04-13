"""PaperSearch node - search papers via academic APIs."""
import json
import logging
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)
from typing import Optional
from typing_extensions import override
from comfy_api.latest import io


def _save_papers_to_project(papers: list, project_id: str) -> list:
    """Save papers to asset library and return paper IDs."""
    import os
    saved_ids = []
    base_url = os.getenv("RESEARCH_API_BASE_URL", "http://127.0.0.1:8003") + "/research/papers/"

    for paper in papers:
        paper_data = {
            "title": paper.get("title", ""),
            "authors_text": ", ".join(paper.get("authors", [])),
            "abstract": paper.get("abstract", ""),
            "year": str(paper.get("year", "")) if paper.get("year") else "",
            "venue": paper.get("venue", ""),
            "source": "semantic_scholar",
            "external_id": paper.get("paper_id", ""),
            "library_status": "pending",  # Will be promoted to library manually
            "project_id": project_id if project_id else None,
        }

        try:
            data = json.dumps(paper_data).encode()
            req = urllib.request.Request(
                base_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                saved_ids.append(result.get("id", ""))
        except Exception as e:
            logger.warning(f"Failed to save paper {paper.get('title', 'unknown')}: {e}")

    return saved_ids


class PaperSearch(io.ComfyNode):
    """Search academic papers from Semantic Scholar."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PaperSearch",
            display_name="Paper Search",
            category="Research",
            inputs=[
                io.String.Input(
                    "query",
                    display_name="Search Query",
                    default="",
                ),
                io.Int.Input(
                    "max_results",
                    display_name="Max Results",
                    default=5,
                    min=1,
                    max=20,
                    step=1,
                ),
                io.String.Input(
                    "target_project_id",
                    display_name="Target Project ID",
                    default="",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Papers (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, query: str, max_results: int, target_project_id: str = "") -> io.NodeOutput:
        if not query.strip():
            return io.NodeOutput(papers=json.dumps([]))

        encoded_query = urllib.parse.quote(query)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={max_results}&fields=title,authors,abstract,year,journal,venue"

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
                papers = []
                for item in data.get("data", []):
                    papers.append({
                        "title": item.get("title", ""),
                        "authors": [a.get("name", "") for a in item.get("authors", [])],
                        "abstract": item.get("abstract", ""),
                        "year": item.get("year", ""),
                        "venue": item.get("venue", ""),
                        "paper_id": item.get("paperId", ""),
                    })

                # Save to project if specified
                saved_ids = []
                if target_project_id:
                    saved_ids = _save_papers_to_project(papers, target_project_id)

                result = {
                    "papers": papers,
                    "saved_ids": saved_ids,
                    "saved_count": len(saved_ids),
                }
                return io.NodeOutput(papers=json.dumps(result, indent=2))
        except Exception as e:
            return io.NodeOutput(papers=json.dumps({"error": str(e)}))
