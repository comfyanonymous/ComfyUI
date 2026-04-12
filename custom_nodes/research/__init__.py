"""Research Workbench custom nodes for ComfyUI."""
from typing import List
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class ResearchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        from custom_nodes.research.paper_search import PaperSearch
        from custom_nodes.research.claim_extract import PaperClaimExtract
        from custom_nodes.research.evidence_assemble import ClaimEvidenceAssemble
        return [PaperSearch, PaperClaimExtract, ClaimEvidenceAssemble]


async def comfy_entrypoint() -> ComfyExtension:
    return ResearchExtension()
