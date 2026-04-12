"""PaperSearch node - search papers via academic APIs."""
import json
import urllib.request
import urllib.parse
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


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
            ],
            outputs=[
                io.String.Output(display_name="Papers (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, query: str, max_results: int) -> io.NodeOutput:
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
                return io.NodeOutput(papers=json.dumps(papers, indent=2))
        except Exception as e:
            return io.NodeOutput(papers=json.dumps({"error": str(e)}))
